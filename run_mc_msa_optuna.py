import os
import re
import json
import argparse
import gc
import sys
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt
import optuna

# Setup path to import src modules
sys.path.append(str(Path(__file__).parent.absolute()))

from src.melody_analysis_v2 import MelodyAnalyzer, MelodyClassifierPaper, MelodyFeatures, MelodySegmentAnnotation, DiagramExporter
from src.melody_analysis_v2.classifier_paper import calculate_lcs
from src.melody_analysis_v2.segmenter import MelodySegment
from src.melody_analysis_v2.pipeline import MelodyAnalysisResult
from src.melody_analysis_v2.visualization import (
    plot_boundary_detection, 
    plot_self_similarity, 
    plot_spectrogram_with_segments
)

# Import utilities from the original MC-MSA script
from run_mc_msa import (
    METHOD_CLASSIFICATION,
    get_audio_files,
    pair_files_fuzzy,
    calculate_levenshtein_similarity,
    calculate_pitch_histogram_similarity,
    evaluate_binary_classification,
    load_or_analyze_light,
    load_or_analyze,
    get_audio_metadata,
    plot_caplin_bands,
    plot_caplin_contour,
    plot_contour_only_comparison,
    plot_energy_only_comparison,
    plot_melody_and_energy_comparison,
    find_available_datasets
)

def run_single_dataset_optuna_mc_msa(dataset_dir: Path, methods: list, args, base_dir: Path, cache_dir: Path):
    orig_dir = dataset_dir / args.orig_subdir
    cover_dir = dataset_dir / args.cover_subdir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = dataset_dir / output_dir
        
    if not orig_dir.exists() or not cover_dir.exists():
        print(f"Las carpetas de origen '{orig_dir}' y/o '{cover_dir}' no existen en {dataset_dir.name}. Saltando...")
        return
        
    orig_files = get_audio_files(orig_dir, match_mode=args.match_mode)
    cover_files = get_audio_files(cover_dir, match_mode=args.match_mode)
    
    if args.match_mode == "fuzzy":
        orig_files, cover_files = pair_files_fuzzy(orig_files, cover_files)
        common_ids = sorted(list(orig_files.keys()))
    else:
        common_ids = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
        
    print("\n" + "="*80)
    print(f" PROCESANDO DATASET (CON OPTUNA): {dataset_dir.name} ({len(common_ids)} pares encontrados)")
    print("="*80)
    
    if not common_ids:
        print("No se encontraron pares válidos. Saltando...")
        return

    print("Métodos a evaluar:")
    for m in methods:
        classification = METHOD_CLASSIFICATION.get(m, "Unknown")
        print(f"  - {m}: {classification}")

    # Initialize with default classifier to load and populate cache
    default_classifier = MelodyClassifierPaper()
    
    summary_path = output_dir / "mc_msa_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if summary_path exists but has old format, delete it to avoid column mismatch
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                first_line = f.readline().strip()
            if "mr" not in first_line or "mdr" not in first_line:
                summary_path.unlink()
        except Exception:
            pass

    if not summary_path.exists():
        with open(summary_path, 'w') as f:
            f.write("metodo,pares,lcs_promedio,mr,mrr,mdr,map,top5_prec,top10_prec,dtw_promedio,min_voicing_thresh,slope_epsilon,energy_tau\n")

    for method in methods:
        classification = METHOD_CLASSIFICATION.get(method, "Unknown")
        print(f"\n[{method}] Cargando/Poblando caché... ({classification})")
        analyzer = MelodyAnalyzer(extraction_method=method, classifier=default_classifier)
        
        out_method_dir = output_dir / method
        out_method_dir.mkdir(parents=True, exist_ok=True)

        res_originals = {}
        total_p = len(common_ids)
        for i, uid in enumerate(common_ids, 1):
            try:
                file_path = orig_files[uid]
                prefix = f"  [{i}/{total_p}] ({i/total_p:.1%}) [Original] [{method}]"
                cache_dir_method = cache_dir / method
                tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
                if tiny_path.exists():
                    print(f"{prefix} [Caché] {file_path.name}")
                else:
                    print(f"{prefix} [Procesando] {file_path.name}...")
                res_originals[uid] = load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=prefix)
                gc.collect()
            except Exception as e:
                print(f"\nError analizando original {uid} ({method}): {e}")
                res_originals[uid] = None
        print(f"  Originales cargados y cacheados.")

        res_covers = {}
        for i, uid in enumerate(common_ids, 1):
            try:
                file_path = cover_files[uid]
                prefix = f"  [{i}/{total_p}] ({i/total_p:.1%}) [Cover] [{method}]"
                cache_dir_method = cache_dir / method
                tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
                if tiny_path.exists():
                    print(f"{prefix} [Caché] {file_path.name}")
                else:
                    print(f"{prefix} [Procesando] {file_path.name}...")
                res_covers[uid] = load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=prefix)
                gc.collect()
            except Exception as e:
                print(f"\nError analizando cover {uid} ({method}): {e}")
                res_covers[uid] = None
        print(f"  Covers cargados y cacheados.")

        # --- OPTUNA PHASE: LIGHTWEIGHT IN-MEMORY LOADING ---
        print("\nCargando características y segmentos desde la caché JSON para Optuna...")
        optuna_originals = {}
        optuna_covers = {}
        
        for uid in common_ids:
            if res_originals[uid] is not None:
                file_path = orig_files[uid]
                try:
                    res = load_or_analyze(analyzer, file_path, method, cache_dir)
                    segments = [s.segment for s in res.segments]
                    optuna_originals[uid] = {
                        'features': res.features,
                        'segments': segments
                    }
                except Exception as e:
                    print(f"  Error al cargar original {uid} para Optuna: {e}")
                    optuna_originals[uid] = None
            else:
                optuna_originals[uid] = None
                
            if res_covers[uid] is not None:
                file_path = cover_files[uid]
                try:
                    res = load_or_analyze(analyzer, file_path, method, cache_dir)
                    segments = [s.segment for s in res.segments]
                    optuna_covers[uid] = {
                        'features': res.features,
                        'segments': segments
                    }
                except Exception as e:
                    print(f"  Error al cargar cover {uid} para Optuna: {e}")
                    optuna_covers[uid] = None
            else:
                optuna_covers[uid] = None

        print("Datos cargados en memoria. Iniciando optimización con Optuna...")
        
        # Optuna options
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            min_voicing_thresh = trial.suggest_float('min_voicing_thresh', 0.0, 0.8)
            slope_epsilon = trial.suggest_float('slope_epsilon', 0.0, 0.5)
            energy_tau = trial.suggest_float('energy_tau', 0.01, 0.9)
            
            clf = MelodyClassifierPaper(
                min_voicing_thresh=min_voicing_thresh,
                slope_epsilon=slope_epsilon,
                energy_tau=energy_tau
            )
            
            seq_originals = {}
            for uid, data in optuna_originals.items():
                if data is None: continue
                annotations = clf.classify(data['features'], data['segments'])
                seq_originals[uid] = [ann.label for ann in annotations]
                
            seq_covers = {}
            for uid, data in optuna_covers.items():
                if data is None: continue
                annotations = clf.classify(data['features'], data['segments'])
                seq_covers[uid] = [ann.label for ann in annotations]
                
            if args.optuna_metric == 'mrr':
                mrr_sum = 0.0
                valid_count = 0
                for uid_cover, seq_cov in seq_covers.items():
                    similarities = []
                    for uid_orig, seq_orig in seq_originals.items():
                        sim = calculate_lcs(seq_orig, seq_cov)
                        similarities.append((sim, uid_orig))
                    if not similarities: continue
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    
                    rank = -1
                    for idx, (sim, r_uid) in enumerate(similarities):
                        if r_uid == uid_cover:
                            rank = idx + 1
                            break
                    if rank != -1:
                        valid_count += 1
                        mrr_sum += 1.0 / rank
                return mrr_sum / valid_count if valid_count > 0 else 0.0
                
            else: # 'lcs'
                lcs_sum = 0.0
                valid_count = 0
                for uid, seq_cov in seq_covers.items():
                    if uid in seq_originals:
                        sim = calculate_lcs(seq_originals[uid], seq_cov)
                        lcs_sum += sim
                        valid_count += 1
                return lcs_sum / valid_count if valid_count > 0 else 0.0

        study.optimize(objective, n_trials=args.optuna_trials)
        best_params = study.best_params
        best_value = study.best_value
        
        print("\n" + "*"*60)
        print(f" OPTIMIZACIÓN COMPLETADA PARA EL MÉTODO: {method.upper()}")
        print(f" Mejor {args.optuna_metric.upper()} en entrenamiento: {best_value:.4f}")
        print(" Parámetros óptimos:")
        for k, v in best_params.items():
            print(f"   - {k}: {v:.6f}")
        print("*"*60 + "\n")
        
        # Instantiate the optimized classifier
        best_clf = MelodyClassifierPaper(
            min_voicing_thresh=best_params['min_voicing_thresh'],
            slope_epsilon=best_params['slope_epsilon'],
            energy_tau=best_params['energy_tau']
        )
        
        # Re-classify results in memory for final evaluation
        for uid in common_ids:
            if res_originals[uid] is not None and optuna_originals[uid] is not None:
                ann_orig = best_clf.classify(optuna_originals[uid]['features'], optuna_originals[uid]['segments'])
                res_originals[uid]['seq'] = [ann.label for ann in ann_orig]
                
            if res_covers[uid] is not None and optuna_covers[uid] is not None:
                ann_cover = best_clf.classify(optuna_covers[uid]['features'], optuna_covers[uid]['segments'])
                res_covers[uid]['seq'] = [ann.label for ann in ann_cover]

        # Free optuna memory before evaluating
        del optuna_originals, optuna_covers
        gc.collect()

        # --- FINAL EVALUATION OF RESULTS WITH OPTIMAL PARAMETERS ---
        print("Ejecutando evaluación final con los parámetros optimizados...")
        
        lcs_list, dtw_list, mrr_sum, top5_hits, top10_hits, valid_count = [], [], 0.0, 0, 0, 0
        ranks_list = []
        best_lcs, best_uid = -1.0, None
        detailed_results = []
        
        pairwise_lcs = []
        pairwise_lev = []
        pairwise_pitch_hist = []
        pairwise_dtw = []
        all_comparisons = []

        # Load comparisons cache if it exists
        comp_cache_path = cache_dir / method / f"comparison_cache_{dataset_dir.name}_optuna.json"
        comp_cache = {}
        comp_cache_changed = False
        if comp_cache_path.exists():
            try:
                with open(comp_cache_path, 'r', encoding='utf-8') as f:
                    comp_cache = json.load(f)
                print(f"  [Caché] Cargadas comparaciones previas desde {comp_cache_path.name}")
            except Exception as e:
                print(f"  [Caché] Advertencia al cargar caché de comparaciones: {e}")
        
        for i, uid_cover in enumerate(common_ids, 1):
            try:
                print(f"  [{i}/{total_p}] ({i/total_p:.1%}) Comparando cover: ID {uid_cover}...", end='\r')
                if res_covers[uid_cover] is None: continue
                
                seq_cover = res_covers[uid_cover]['seq']
                pitch_m = res_covers[uid_cover]['pitch_midi']
                
                if pitch_m is None or len(pitch_m) == 0: continue
                
                f0_cover = np.nan_to_num(np.where(pitch_m > 0, 440.0 * np.power(2.0, (pitch_m - 69.0) / 12.0), 0))
                
                similarities = []
                for uid_orig in common_ids:
                    if res_originals[uid_orig] is None: continue
                    seq_orig = res_originals[uid_orig]['seq']
                    pitch_o = res_originals[uid_orig]['pitch_midi']
                    
                    key = f"{orig_files[uid_orig].name}:::{cover_files[uid_cover].name}"
                    cached_entry = comp_cache.get(key, {})
                    
                    import hashlib
                    orig_repr = ",".join(seq_orig) + f"|len:{len(pitch_o)}"
                    cover_repr = ",".join(seq_cover) + f"|len:{len(pitch_m)}"
                    h = hashlib.md5(f"{orig_repr}:::{cover_repr}".encode('utf-8')).hexdigest()
                    
                    if cached_entry.get("hash") != h:
                        cached_entry = {"hash": h}
                        
                    cache_updated = False
                    
                    # LCS
                    if "lcs_similarity" in cached_entry:
                        lcs_sim = cached_entry["lcs_similarity"]
                    else:
                        lcs_sim = calculate_lcs(seq_orig, seq_cover)
                        cached_entry["lcs_similarity"] = lcs_sim
                        cache_updated = True
                        
                    # Levenshtein
                    if "levenshtein_similarity" in cached_entry:
                        lev_sim = cached_entry["levenshtein_similarity"]
                    else:
                        lev_sim = calculate_levenshtein_similarity(seq_orig, seq_cover)
                        cached_entry["levenshtein_similarity"] = lev_sim
                        cache_updated = True
                        
                    # Pitch Histogram
                    if "pitch_hist_similarity" in cached_entry:
                        pitch_hist_sim = cached_entry["pitch_hist_similarity"]
                    else:
                        pitch_hist_sim = calculate_pitch_histogram_similarity(pitch_o, pitch_m)
                        cached_entry["pitch_hist_similarity"] = pitch_hist_sim
                        cache_updated = True
                        
                    # DTW
                    dtw_val = -1.0
                    is_correct = (uid_orig == uid_cover)
                    if args.dtw_all_pairs or is_correct:
                        if "dtw_distance" in cached_entry and cached_entry["dtw_distance"] != "":
                            dtw_val = cached_entry["dtw_distance"]
                        else:
                            f0_orig = np.nan_to_num(np.where(pitch_o > 0, 440.0 * np.power(2.0, (pitch_o - 69.0) / 12.0), 0))
                            try:
                                if len(f0_cover) > 38760:
                                    ds_factor = len(f0_cover) // 1000
                                    f0_cover_ds = f0_cover[::ds_factor]
                                    f0_orig_ds = f0_orig[::ds_factor]
                                  
                                else:
                                    f0_cover_ds = f0_cover
                                    f0_orig_ds = f0_orig
                                    
                                D, wp = librosa.sequence.dtw(f0_cover_ds.reshape(1, -1), f0_orig_ds.reshape(1, -1))
                                dtw_val = D[-1, -1] / len(wp)
                            except:
                                dtw_val = -1.0
                            cached_entry["dtw_distance"] = dtw_val if dtw_val >= 0 else ""
                            cache_updated = True
                            
                    if cache_updated:
                        comp_cache[key] = cached_entry
                        comp_cache_changed = True
                            
                    pairwise_lcs.append((lcs_sim, is_correct))
                    pairwise_lev.append((lev_sim, is_correct))
                    pairwise_pitch_hist.append((pitch_hist_sim, is_correct))
                    if dtw_val >= 0:
                        pairwise_dtw.append((dtw_val, is_correct))
                        
                    all_comparisons.append({
                        "cover_id": uid_cover,
                        "original_id": uid_orig,
                        "lcs_similarity": lcs_sim,
                        "levenshtein_similarity": lev_sim,
                        "pitch_hist_similarity": pitch_hist_sim,
                        "dtw_distance": dtw_val if dtw_val >= 0 else "",
                        "is_correct": 1 if is_correct else 0
                    })
                    
                    similarities.append((lcs_sim, uid_orig))
                
                if not similarities: continue
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                rank, true_sim = -1, 0.0
                for idx, (sim, r_uid) in enumerate(similarities):
                    if r_uid == uid_cover:
                        rank, true_sim = idx + 1, sim
                        break
                
                if rank != -1:
                    valid_count += 1
                    ranks_list.append(rank)
                    mrr_sum += 1.0 / rank
                    if rank <= 5: top5_hits += 1
                    if rank <= 10: top10_hits += 1
                    lcs_list.append(true_sim)
                    
                    if true_sim > best_lcs:
                        best_lcs, best_uid = true_sim, uid_cover
                    
                    correct_dtw = -1.0
                    for comp in all_comparisons:
                        if comp["cover_id"] == uid_cover and comp["original_id"] == uid_cover:
                            if comp["dtw_distance"] != "":
                                correct_dtw = comp["dtw_distance"]
                            break
                    if correct_dtw >= 0:
                        dtw_list.append(correct_dtw)
                    
                    id_label = f"ID {uid_cover:02d}" if isinstance(uid_cover, int) else f"ID {uid_cover}"
                    detailed_results.append(f"{id_label} | LCS: {true_sim:.4f} | Rank: {rank:2d} | DTW: {correct_dtw:.4f}")
            except Exception as e:
                print(f"\nError procesando cover {uid_cover} ({method}): {e}")
            finally:
                if 'res_cover' in locals():
                    del res_cover
                if i % 10 == 0:
                    gc.collect()
                    
        # Save comparisons cache if there were changes
        if comp_cache_changed:
            try:
                comp_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(comp_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(comp_cache, f, indent=2)
                print(f"\n  [Caché] Comparaciones guardadas/actualizadas en {comp_cache_path.name}")
            except Exception as e:
                print(f"\n  [Caché] Advertencia al guardar caché de comparaciones: {e}")

        # Summary metrics
        print(f"\n[{method}] Finalizado.")
        avg_lcs = np.mean(lcs_list) if lcs_list else 0
        mr = np.mean(ranks_list) if ranks_list else 0
        mrr = mrr_sum / valid_count if valid_count else 0
        mdr = np.median(ranks_list) if ranks_list else 0
        map_val = np.mean([1.0 / r for r in ranks_list]) if ranks_list else 0
        top5_prec = top5_hits / valid_count if valid_count else 0
        top10_prec = top10_hits / valid_count if valid_count else 0
        avg_dtw = np.mean(dtw_list) if dtw_list else 0
        
        print(f"[{method}] Optimal Results | LCS: {avg_lcs:.4f} | MR: {mr:.2f} | MRR: {mrr:.4f} | MDR: {mdr:.1f} | MAP: {map_val:.4f} | Top5: {top5_prec:.2%} | Top10: {top10_prec:.2%} | DTW: {avg_dtw:.4f}")
        
        # Export to summary CSV
        with open(summary_path, 'a') as f:
            f.write(f"{method}_optuna,{valid_count},{avg_lcs:.6f},{mr:.6f},{mrr:.6f},{mdr:.1f},{map_val:.6f},{top5_prec:.6f},{top10_prec:.6f},{avg_dtw:.6f},{best_params['min_voicing_thresh']:.6f},{best_params['slope_epsilon']:.6f},{best_params['energy_tau']:.6f}\n")
            
        # Evaluate binary classification and optimal thresholds
        best_thresh_lcs, best_metrics_lcs, curves_lcs = evaluate_binary_classification(pairwise_lcs, "LCS")
        best_thresh_lev, best_metrics_lev, curves_lev = evaluate_binary_classification(pairwise_lev, "Levenshtein")
        best_thresh_ph, best_metrics_ph, curves_ph = evaluate_binary_classification(pairwise_pitch_hist, "Pitch Histogram")
        best_thresh_dtw, best_metrics_dtw, curves_dtw = evaluate_binary_classification(pairwise_dtw, "DTW", lower_is_better=True)
        
        # Export all_comparisons.csv
        comp_csv_path = out_method_dir / "all_comparisons.csv"
        with open(comp_csv_path, 'w') as f:
            f.write("cover_id,original_id,lcs_similarity,levenshtein_similarity,pitch_hist_similarity,dtw_distance,is_correct\n")
            for comp in all_comparisons:
                f.write(f"{comp['cover_id']},{comp['original_id']},{comp['lcs_similarity']:.6f},{comp['levenshtein_similarity']:.6f},{comp['pitch_hist_similarity']:.6f},{comp['dtw_distance']},{comp['is_correct']}\n")
                
        # Export threshold curves
        for m_name, curves in [("lcs", curves_lcs), ("levenshtein", curves_lev), ("pitch_hist", curves_ph), ("dtw", curves_dtw)]:
            if not curves: continue
            curve_csv_path = out_method_dir / f"threshold_analysis_{m_name}.csv"
            with open(curve_csv_path, 'w') as f:
                f.write("threshold,tp,fp,fn,tn,precision,recall,f1_score,accuracy\n")
                for c in curves:
                    f.write(f"{c['threshold']:.4f},{c['tp']},{c['fp']},{c['fn']},{c['tn']},{c['precision']:.6f},{c['recall']:.6f},{c['f1_score']:.6f},{c['accuracy']:.6f}\n")
        
        # Export detailed report TXT
        report_path = out_method_dir / "detailed_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"DETAILED REPORT (OPTIMIZED WITH OPTUNA) - METHOD: {method}\n")
            f.write("="*50 + "\n")
            f.write(f"OPTIMAL CLASSIFIER PARAMETERS:\n")
            for k, v in best_params.items():
                f.write(f"  - {k}: {v:.6f}\n")
            f.write(f"Objective metric ({args.optuna_metric.upper()}): {best_value:.4f}\n")
            f.write("="*50 + "\n")
            if best_uid is not None:
                f.write(f"IMAGES GENERATED FOR THE BEST MATCH (LCS = {best_lcs:.4f}):\n")
                f.write(f"  ID: {best_uid}\n")
                f.write(f"  Original: {orig_files[best_uid].name}\n")
                f.write(f"  Cover:    {cover_files[best_uid].name}\n")
                f.write("="*50 + "\n")
            f.write("\n".join(detailed_results) + "\n")
            f.write("="*50 + "\n")
            f.write(f"GENERAL SUMMARY:\n")
            f.write(f"Pairs evaluated: {valid_count}\n")
            f.write(f"Average LCS:    {avg_lcs:.4f}\n")
            f.write(f"Mean Rank:      {mr:.4f}\n")
            f.write(f"MRR:             {mrr:.4f}\n")
            f.write(f"Median Rank:     {mdr:.1f}\n")
            f.write(f"MAP:             {map_val:.4f}\n")
            f.write(f"Top-5 Precision: {top5_prec:.2%}\n")
            f.write(f"Top-10 Precision: {top10_prec:.2%}\n")
            f.write(f"Average DTW:    {avg_dtw:.4f}\n")
            f.write("="*50 + "\n")
            f.write(f"BINARY CLASSIFICATION THRESHOLDS ANALYSIS (OPTIMIZING F1-SCORE):\n\n")
            
            for m_name, best_t, best_m in [
                ("LCS (Longest Common Subsequence)", best_thresh_lcs, best_metrics_lcs),
                ("Levenshtein (Edit Distance)", best_thresh_lev, best_metrics_lev),
                ("Pitch Class Histogram (Chroma Cosine)", best_thresh_ph, best_metrics_ph),
                ("DTW Distance (Optimal Path)", best_thresh_dtw, best_metrics_dtw)
            ]:
                f.write(f"--- Metrica: {m_name} ---\n")
                if best_m:
                    f.write(f"  Optimal Threshold:  {best_t:.4f}\n")
                    f.write(f"  F1-Score:       {best_m['f1_score']:.4f}\n")
                    f.write(f"  Precision:      {best_m['precision']:.4f}\n")
                    f.write(f"  Recall (Sens.): {best_m['recall']:.4f}\n")
                    f.write(f"  Accuracy:       {best_m['accuracy']:.4f}\n")
                    f.write(f"  Confusion Matrix:\n")
                    f.write(f"    - TP (True Pos.):  {best_m['tp']}\n")
                    f.write(f"    - FP (False Pos.): {best_m['fp']}\n")
                    f.write(f"    - FN (False Neg.): {best_m['fn']}\n")
                    f.write(f"    - TN (True Neg.):  {best_m['tn']}\n")
                else:
                    f.write(f"  Not enough data to evaluate.\n")
                f.write("\n")
            f.write("="*50 + "\n")

        # Re-initialize primary analyzer with best classifier for plotting
        analyzer = MelodyAnalyzer(extraction_method=method, classifier=best_clf)

        # Qualitative plots for the best match of this method
        if best_uid is not None:
            print(f"\n[{method}] Generando gráficas cualitativas finales usando el clasificador optimizado (Mejor Match ID {best_uid})...")
            try:
                from src.melody_analysis_v2.visualization import (
                    plot_melspectrogram, plot_melody_contour, plot_melody_only,
                    plot_energy_only, plot_melody_and_energy
                )
                
                meta_orig = get_audio_metadata(orig_files[best_uid])
                meta_cover = get_audio_metadata(cover_files[best_uid])
                title_orig = f"{meta_orig[0]} ({meta_orig[1]})"
                title_cover = f"{meta_cover[0]} ({meta_cover[1]})"
                
                print(f"  Procesando Original (ID {best_uid})...")
                res_orig_best = analyzer.analyze_file(str(orig_files[best_uid]))
                
                # Novelty, SSM, Contours
                plot_boundary_detection(res_orig_best, output_path=out_method_dir / "fig_novelty_orig.png", title=f"Boundary Detection (Original)\nSong: {title_orig}")
                if res_orig_best.self_similarity is not None:
                    plot_self_similarity(res_orig_best, output_path=out_method_dir / "fig_ssm_orig.png", title=f"SSM (Original)\nSong: {title_orig}")
                
                plot_melody_contour(res_orig_best, output_path=out_method_dir / "fig_contour_orig.png", title=f"Melodic Contour (Original)\nSong: {title_orig}")
                plot_melody_only(res_orig_best, output_path=out_method_dir / "fig_contour_only_orig.png", show_segments=False, title=f"Melodic Contour Only (Original)\nSong: {title_orig}")
                plot_energy_only(res_orig_best, output_path=out_method_dir / "fig_energy_only_orig.png", title=f"Normalized Energy Only (Original)\nSong: {title_orig}")
                plot_melody_and_energy(res_orig_best, output_path=out_method_dir / "fig_contour_and_energy_orig.png", title=f"Melodic Contour & Energy (Original)\nSong: {title_orig}")
                
                try:
                    audio_plot, sr_plot = librosa.load(orig_files[best_uid], sr=analyzer.sample_rate)
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_orig_best, output_path=out_method_dir / "fig_spectrogram_orig.png", title=f"Spectrogram with Segments (Original)\nSong: {title_orig}")
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_orig.png", title=f"Mel-spectrogram (Original)\nSong: {title_orig}")
                    del audio_plot
                except Exception as spec_err:
                    print(f"  Advertencia graficando espectrograma (orig): {spec_err}")
                
                plt.close('all')
                
                print(f"  Procesando Cover (ID {best_uid})...")
                res_cover_best = analyzer.analyze_file(str(cover_files[best_uid]))
                
                # Novelty, SSM, Contours
                plot_boundary_detection(res_cover_best, output_path=out_method_dir / "fig_novelty_cover.png", title=f"Boundary Detection (Cover)\nSong: {title_cover}")
                if res_cover_best.self_similarity is not None:
                    plot_self_similarity(res_cover_best, output_path=out_method_dir / "fig_ssm_cover.png", title=f"SSM (Cover)\nSong: {title_cover}")
                
                plot_melody_contour(res_cover_best, output_path=out_method_dir / "fig_contour_cover.png", title=f"Melodic Contour (Cover)\nSong: {title_cover}")
                plot_melody_only(res_cover_best, output_path=out_method_dir / "fig_contour_only_cover.png", show_segments=False, title=f"Melodic Contour Only (Cover)\nSong: {title_cover}")
                plot_energy_only(res_cover_best, output_path=out_method_dir / "fig_energy_only_cover.png", title=f"Normalized Energy Only (Cover)\nSong: {title_cover}")
                plot_melody_and_energy(res_cover_best, output_path=out_method_dir / "fig_contour_and_energy_cover.png", title=f"Melodic Contour & Energy (Cover)\nSong: {title_cover}")
                
                try:
                    audio_plot, sr_plot = librosa.load(cover_files[best_uid], sr=analyzer.sample_rate)
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_cover_best, output_path=out_method_dir / "fig_spectrogram_cover.png", title=f"Spectrogram with Segments (Cover)\nSong: {title_cover}")
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_cover.png", title=f"Mel-spectrogram (Cover)\nSong: {title_cover}")
                    del audio_plot
                except Exception as spec_err:
                    print(f"  Advertencia graficando espectrograma (cover): {spec_err}")
                
                plt.close('all')
                
                # Shared Plots (Caplin Bands and Contours)
                plot_caplin_bands(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_bands.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_caplin_contour(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_contour_only_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour_only.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_energy_only_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_energy_only.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_melody_and_energy_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour_and_energy.png", meta_orig=meta_orig, meta_cover=meta_cover)
                
                # Export 9 diagram steps for Best Match
                print(f"  Exportando 9 pasos del diagrama para el Mejor Match (ID {best_uid})...")
                exporter_orig = DiagramExporter(out_method_dir / "diagrama_pasos_original")
                exporter_orig.export_all(str(orig_files[best_uid]), method=method)
                
                exporter_cover = DiagramExporter(out_method_dir / "diagrama_pasos_cover")
                exporter_cover.export_all(str(cover_files[best_uid]), method=method)
                
                print(f"  Gráficas generadas exitosamente en {out_method_dir}")
                
                del res_orig_best, res_cover_best
                gc.collect()
            except Exception as plot_err:
                print(f"  Error al generar gráficas: {plot_err}")
            finally:
                plt.close('all')
                gc.collect()

    # Save consolidated comparative table
    save_dataset_comparative_table(dataset_dir, output_dir)

def save_dataset_comparative_table(dataset_dir: Path, output_dir: Path):
    summary_path = output_dir / "mc_msa_summary.csv"
    if not summary_path.exists():
        print(f"[Comparative Table] Summary file not found at {summary_path}")
        return
        
    dataset_name = dataset_dir.name
    
    # Read rows from summary_path and keep only the latest row per method
    method_rows = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            header_indices = {col.strip().lower(): i for i, col in enumerate(header)}
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > 0:
                    method_rows[parts[0]] = parts
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return
        
    if not method_rows:
        return
        
    sorted_methods = sorted(list(method_rows.keys()))
    
    # We want to format columns dynamically based on header presence
    has_mr = "mr" in header_indices
    has_mdr = "mdr" in header_indices
    has_map = "map" in header_indices
    has_top10 = "top10_prec" in header_indices or "top10" in header_indices
    has_opt_params = "min_voicing_thresh" in header_indices
    
    lines = []
    divider_len = 160 if has_opt_params else 120
    lines.append("-" * divider_len)
    lines.append(f"Dataset: {dataset_name} (Optimizado con Optuna)")
    lines.append("-" * divider_len)
    
    # Header line
    hdr_cols = [f"{'Method':<25}", f"{'Avg. LCS (%)':<14}"]
    if has_mr: hdr_cols.append(f"{'MR':<8}")
    hdr_cols.append(f"{'MRR (%)':<10}")
    if has_mdr: hdr_cols.append(f"{'MDR':<8}")
    if has_map: hdr_cols.append(f"{'MAP (%)':<10}")
    hdr_cols.append(f"{'Top-5 (%)':<12}")
    if has_top10: hdr_cols.append(f"{'Top-10 (%)':<12}")
    hdr_cols.append(f"{'DTW':<10}")
    if has_opt_params:
        hdr_cols.append(f"{'Optimized Parameters (voicing, slope, energy)':<45}")
    lines.append(" | ".join(hdr_cols))
    lines.append("-" * divider_len)
    
    for method in sorted_methods:
        row = method_rows[method]
        method_disp = method.upper()
        
        def get_val(name, is_pct=False, fmt=".2f", default="-"):
            idx = header_indices.get(name.lower())
            if idx is not None and idx < len(row) and row[idx] != "":
                try:
                    val = float(row[idx])
                    if is_pct:
                        val *= 100
                    return f"{val:{fmt}}"
                except ValueError:
                    return row[idx]
            return default

        try:
            lcs = get_val("lcs_promedio" if "lcs_promedio" in header_indices else "avg_lcs", is_pct=True) + "%"
            mrr = get_val("mrr", is_pct=True) + "%"
            top5 = get_val("top5_prec", is_pct=True) + "%"
            dtw = get_val("dtw_promedio" if "dtw_promedio" in header_indices else "avg_dtw")
            
            row_cols = [f"{method_disp:<25}", f"{lcs:>12}"]
            if has_mr:
                mr = get_val("mr")
                row_cols.append(f"{mr:>8}")
            row_cols.append(f"{mrr:>8}")
            if has_mdr:
                mdr = get_val("mdr", fmt=".1f")
                row_cols.append(f"{mdr:>8}")
            if has_map:
                map_val = get_val("map", is_pct=True) + "%"
                row_cols.append(f"{map_val:>8}")
            row_cols.append(f"{top5:>10}")
            if has_top10:
                top10 = get_val("top10_prec" if "top10_prec" in header_indices else "top10", is_pct=True) + "%"
                row_cols.append(f"{top10:>10}")
            row_cols.append(f"{dtw:>10}")
            
            if has_opt_params:
                v_t = get_val("min_voicing_thresh", fmt=".4f")
                s_e = get_val("slope_epsilon", fmt=".4f")
                e_t = get_val("energy_tau", fmt=".4f")
                if v_t != "-" and s_e != "-" and e_t != "-":
                    params_str = f"voicing: {v_t}, slope: {s_e}, energy: {e_t}"
                else:
                    params_str = "N/A"
                row_cols.append(f"{params_str:<45}")
                
            lines.append(" | ".join(row_cols))
        except Exception as e:
            lines.append(f"{method.upper():<25} | Error formatting row: {e}")
            
    lines.append("-" * divider_len)
    
    table_content = "\n".join(lines) + "\n"
    table_path = dataset_dir / "comparative_table_optuna.txt"
    try:
        table_path.write_text(table_content, encoding='utf-8')
        print(f"\n[Comparative Table] Successfully saved at {table_path}")
    except Exception as e:
        print(f"Error writing comparative table: {e}")


def main():
    available_methods = [
        'all', 'all_f0', 'all_melody',
        'pyin', 'yin', 'crepe', 'rmvpe', 'spice', 'jdc', 'fcn_f0',
        'melodia', 'tachibana', 'poliner', 'durrieu', 'basic_pitch',
        'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
        'bs_roformer', 'demucs', 'ensemble'
    ]
    parser = argparse.ArgumentParser(description="Evaluación MC-MSA para extracción de melodía optimizando el clasificador con Optuna.")
    parser.add_argument("--method", type=str, default=None, 
                        choices=available_methods,
                        help="Método de extracción a utilizar")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Directorio base del dataset")
    parser.add_argument("--orig_subdir", type=str, default="originales",
                        help="Subdirectorio de canciones originales")
    parser.add_argument("--cover_subdir", type=str, default="covers",
                        help="Subdirectorio de canciones covers")
    parser.add_argument("--output_dir", type=str, default="resultados_mc_msa_optuna",
                        help="Directorio para salidas (resuelto dentro de la carpeta del dataset si es relativo)")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directorio para la caché de análisis JSON")
    parser.add_argument("--match_mode", type=str, default=None,
                        choices=["id", "stem", "fuzzy"],
                        help="Método de emparejamiento")
    parser.add_argument("--dtw_all_pairs", action="store_true",
                        help="Calcular DTW para todos los pares")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Eliminar la caché del método seleccionado antes de comenzar")
    parser.add_argument("--optuna_trials", type=int, default=100,
                        help="Número de ensayos/trials para la optimización de Optuna")
    parser.add_argument("--optuna_metric", type=str, default="mrr",
                        choices=["mrr", "lcs"],
                        help="Métrica a maximizar con Optuna (mrr o lcs)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()

    # Interactive Dataset selection if not defined via CLI
    if args.dataset_dir is None:
        datasets = find_available_datasets(base_dir)
        if not datasets:
            print("\nNo se detectaron carpetas de dataset automáticamente en el directorio base.")
            manual = input("Por favor ingrese la ruta o nombre del dataset a utilizar: ").strip()
            args.dataset_dir = manual
        else:
            print("\n=== Selección de Dataset (Optuna) ===")
            for i, d in enumerate(datasets, 1):
                print(f"{i}. {d}")
            print(f"{len(datasets) + 1}. [Procesar TODOS los datasets de una vez]")
            print(f"{len(datasets) + 2}. [Ingresar otra ruta manual...]")
            
            while True:
                try:
                    choice = input(f"\nSeleccione un dataset (1-{len(datasets) + 2}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(datasets):
                        args.dataset_dir = datasets[idx]
                        break
                    elif idx == len(datasets):
                        args.dataset_dir = "all"
                        break
                    elif idx == len(datasets) + 1:
                        manual = input("Ingrese la ruta o nombre del dataset: ").strip()
                        if manual:
                            args.dataset_dir = manual
                            break
                    else:
                        print(f"Error: Por favor seleccione un número entre 1 y {len(datasets) + 2}.")
                except ValueError:
                    if choice in datasets:
                        args.dataset_dir = choice
                        break
                    elif choice.lower() == "all":
                        args.dataset_dir = "all"
                        break
                    print("Error: Entrada no válida.")

    # Interactive Method selection if not defined via CLI
    if args.method is None:
        print("\n=== Selección de Método de Extracción (Optuna) ===")
        f0_methods = ['pyin', 'yin', 'crepe', 'ensemble', 'rmvpe', 'spice', 'jdc', 'fcn_f0']
        melody_methods = [
            'poliner', 'durrieu', 'tachibana', 'melodia', 'basic_pitch',
            'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
            'bs_roformer', 'demucs'
        ]
        other_methods = []
        
        idx_map = {}
        curr_idx = 1
        
        print("\n--- Extractores de F0 ---")
        for m in f0_methods:
            print(f"  {curr_idx:2d}. {m}")
            idx_map[curr_idx] = m
            curr_idx += 1
        print(f"  {curr_idx:2d}. {'all_f0':<20} [Todos los extractores de F0]")
        idx_map[curr_idx] = 'all_f0'
        curr_idx += 1
        
        print("\n--- Extractores de Melodía ---")
        for m in melody_methods:
            print(f"  {curr_idx:2d}. {m}")
            idx_map[curr_idx] = m
            curr_idx += 1
        print(f"  {curr_idx:2d}. {'all_melody':<20} [Todos los extractores de Melodía]")
        idx_map[curr_idx] = 'all_melody'
        curr_idx += 1
        
        print("\n--- Otros / Especiales ---")
        for m in other_methods:
            classification = METHOD_CLASSIFICATION.get(m, "")
            print(f"  {curr_idx:2d}. {m:<20} [{classification}]")
            idx_map[curr_idx] = m
            curr_idx += 1
        print(f"  {curr_idx:2d}. {'all':<20} [Todos los métodos]")
        idx_map[curr_idx] = 'all'
        
        while True:
            try:
                choice = input(f"\nSeleccione un método (1-{curr_idx}): ").strip()
                if choice.lower() in available_methods:
                    args.method = choice.lower()
                    break
                idx = int(choice)
                if 1 <= idx <= curr_idx:
                    args.method = idx_map[idx]
                    break
                else:
                    print(f"Error: Por favor seleccione un número entre 1 y {curr_idx}.")
            except ValueError:
                if choice.strip().lower() in available_methods:
                    args.method = choice.strip().lower()
                    break
                print("Error: Entrada no válida.")

    # Interactive Matching selection if not defined via CLI
    if args.match_mode is None:
        print("\n=== Selección de Método de Emparejamiento (Optuna) ===")
        print("1. Por ID Numérico (ej: '01 - Pedro Infante.wav' con '01 - Cover.mp3')")
        print("2. Por Nombre / Stem Exacto (ej: 'Te_Vi_Venir_Original.wav' con 'Te Vi Venir (Covers).mp3')")
        print("3. Emparejamiento Inteligente / Fuzzy (Para nombres complejos o música clásica)")
        
        while True:
            choice = input("\nSeleccione el método (1-3) [Por defecto: 1]: ").strip()
            if not choice or choice == "1":
                args.match_mode = "id"
                break
            elif choice == "2":
                args.match_mode = "stem"
                break
            elif choice == "3":
                args.match_mode = "fuzzy"
                break
            else:
                print("Error: Por favor seleccione 1, 2 o 3.")

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir

    if args.method == 'all':
        methods = [
            'pyin', 'yin', 'crepe', 'ensemble', 'rmvpe', 'spice', 'fcn_f0',
            'melodia', 'tachibana', 'poliner', 'durrieu', 'basic_pitch',
            'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
            'bs_roformer', 'demucs'
        ]
    elif args.method == 'all_f0':
        methods = ['pyin', 'yin', 'crepe', 'ensemble', 'rmvpe', 'spice', 'jdc', 'fcn_f0']
    elif args.method == 'all_melody':
        methods = [
            'poliner', 'durrieu', 'tachibana', 'melodia', 'basic_pitch',
            'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
            'bs_roformer', 'demucs'
        ]
    else:
        methods = [args.method]

    # Optional cache clearing
    if args.clear_cache:
        for m in methods:
            method_cache_dir = cache_dir / m
            if method_cache_dir.exists():
                print(f"[Caché] Eliminando caché para '{m}' en {method_cache_dir}...")
                import shutil
                shutil.rmtree(method_cache_dir)
    else:
        any_cache_exists = any((cache_dir / m).exists() and (cache_dir / m).is_dir() and any((cache_dir / m).iterdir()) for m in methods)
        if any_cache_exists:
            ans = input(f"\n¿Desea eliminar la caché existente para los métodos a evaluar antes de comenzar? (s/n): ").strip().lower()
            if ans in ['s', 'si', 'y', 'yes']:
                for m in methods:
                    method_cache_dir = cache_dir / m
                    if method_cache_dir.exists():
                        import shutil
                        shutil.rmtree(method_cache_dir)
                print("[Caché] Eliminación completada.")

    # Determine datasets to process
    if args.dataset_dir == "all":
        datasets_to_process = []
        datasets_names = find_available_datasets(base_dir)
        for name in datasets_names:
            path = Path(name)
            if not path.is_absolute():
                path = base_dir / path
            datasets_to_process.append(path)
    else:
        path = Path(args.dataset_dir)
        if not path.is_absolute():
            path = base_dir / path
        datasets_to_process = [path]

    # Process each dataset
    for dataset_dir in datasets_to_process:
        try:
            run_single_dataset_optuna_mc_msa(dataset_dir, methods, args, base_dir, cache_dir)
        except Exception as e:
            print(f"\nError al procesar dataset '{dataset_dir.name}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
