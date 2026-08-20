#!/usr/bin/env python3
"""
MC-MSA Custom Parameters Empirical Sweep Runner
------------------------------------------------
Evaluates the MC-MSA framework across custom literature-defined/thesis hyperparameters
and saves detailed run results to text/CSV files for subsequent analysis.

Target Hyperparameters:
- L = 8 (Radius of the 2D Gaussian checkerboard kernel)
- σ = 2 (Standard deviation for 1D Gaussian smoothing)
- τ_peak = 0.20 (Minimum peak height threshold relative to global max)
- y = 0.20 (20% final portion of segment used for functional classification)
- θ_slope = ±0.15 (Pitch slope thresholds for detecting ascending/descending contours)
- τ_E = [0.3, 0.15] (Normalized energy thresholds for detecting sustained intensity or drop)

Supports running sweeps across single or multiple energy thresholds while caching
intermediate feature representations for high performance.
"""

import os
import sys
import argparse
import gc
import json
import itertools
from pathlib import Path
import numpy as np

# Ensure root src is in python path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import librosa
from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodySegmenter,
    MelodyClassifierThesis
)
from run_mc_msa import (
    calculate_lcs,
    get_audio_files,
    pair_files_fuzzy,
    find_available_datasets,
    safe_json
)
from src.melody_analysis_v2.segmenter import MelodySegmenter, MelodySegment
from src.melody_analysis_v2.features import MelodyFeatures


def load_features_fast(analyzer, file_path, method, cache_dir):
    method_dir = cache_dir / method
    cache_path = method_dir / f"{file_path.stem}.json"
    
    if not cache_path.exists() and method_dir.exists():
        prefix = file_path.stem.split(" - ")[0] if " - " in file_path.stem else file_path.stem
        matches = [
            p for p in method_dir.glob(f"{prefix}*.json")
            if not p.name.endswith(".tiny.json") and not p.name.startswith("comparison_cache")
        ]
        if matches:
            cache_path = matches[0]

    if cache_path and cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            feat = MelodyFeatures.from_dict(data)
            del data
            return feat, cache_path
        except Exception as e:
            print(f"Warning loading cache for {file_path.name}: {e}, re-analyzing...")
            
    if analyzer is None:
        analyzer = MelodyAnalyzer(extraction_method=method)
    res = analyzer.analyze_file(str(file_path))
    target_p = method_dir / f"{file_path.stem}.json"
    target_p.parent.mkdir(parents=True, exist_ok=True)
    with open(target_p, 'w', encoding='utf-8') as f:
        json.dump(res.to_dict(), f, default=safe_json)
    feat = res.features
    del res
    gc.collect()
    return feat, target_p


def load_or_compute_ssm(feat: MelodyFeatures, file_path: Path, method: str, cache_dir: Path) -> np.ndarray:
    method_dir = cache_dir / method
    ssm_path = method_dir / f"{file_path.stem}.ssm.npy"
    
    if ssm_path.exists():
        try:
            return np.load(ssm_path)
        except Exception:
            pass

    segmenter_base = MelodySegmenter()
    n_frames = len(feat.times)
    if n_frames > segmenter_base.max_ssm_frames:
        step = int(np.ceil(n_frames / segmenter_base.max_ssm_frames))
        ds_f = MelodyFeatures(
            times=feat.times[::step],
            pitch_midi=feat.pitch_midi[::step],
            confidence=feat.confidence[::step],
            energy=feat.energy[::step]
        )
    else:
        ds_f = feat
        
    sim_matrix = segmenter_base.compute_self_similarity(ds_f)
    ssm_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ssm_path, sim_matrix)
    return sim_matrix
    summary_path = output_dir / "mc_msa_summary_custom_params.csv"
    if not summary_path.exists():
        print(f"[Comparative Table] Summary file not found at {summary_path}")
        return
        
    dataset_name = dataset_dir.name
    
    rows = []
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            header = [c.strip().lower() for c in f.readline().strip().split(',')]
            for line in f:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) >= len(header):
                    rows.append(dict(zip(header, parts)))
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return
        
    if not rows:
        return
        
    lines = []
    divider_len = 160
    lines.append("=" * divider_len)
    lines.append(f"TABLA DE RESULTADOS DE EXPERIMENTOS MC-MSA CON PARÁMETROS CUSTOM ({dataset_name})")
    lines.append("=" * divider_len)
    
    hdr_cols = [
        f"{'Method':<18}", f"{'Avg. LCS (%)':<14}", f"{'MRR (%)':<10}", f"{'Top-5 (%)':<10}", 
        f"{'Top-10 (%)':<10}", f"{'DTW':<10}", f"{'Parameters (L, σ, τ_pk, L_tail, ε_slp, τ_E)':<55}"
    ]
    lines.append(" | ".join(hdr_cols))
    lines.append("-" * divider_len)
    
    for r in rows:
        method_disp = r.get("method", "").upper()
        try:
            lcs = f"{float(r.get('avg_lcs', 0))*100:.2f}%"
            mrr = f"{float(r.get('mrr', 0))*100:.2f}%"
            top5 = f"{float(r.get('top5_prec', 0))*100:.2f}%"
            top10 = f"{float(r.get('top10_prec', 0))*100:.2f}%"
            dtw = f"{float(r.get('avg_dtw', 0)):.2f}"
            
            rad = r.get("checkerboard_radius", "-")
            ker = r.get("kernel_size", "-")
            pk = r.get("peak_threshold", "-")
            tl = r.get("tail_proportion", "-")
            s_e = r.get("slope_epsilon", "-")
            e_t = r.get("energy_tau", "-")
            
            params_str = f"L:{rad}, σ:{ker}, τ_pk:{pk}, L_tail:{tl}, ε_slp:{s_e}, τ_E:{e_t}"
            row_cols = [f"{method_disp:<18}", f"{lcs:>14}", f"{mrr:>10}", f"{top5:>10}", f"{top10:>10}", f"{dtw:>10}", f"{params_str:<55}"]
            lines.append(" | ".join(row_cols))
        except Exception as e:
            lines.append(f"{method_disp:<18} | Error formatting row: {e}")
            
    lines.append("=" * divider_len)
    
    # Also calculate and display grand average across all methods if multiple methods exist
    if len(rows) > 1:
        mean_lcs = np.mean([float(r.get('avg_lcs', 0)) for r in rows]) * 100
        mean_mrr = np.mean([float(r.get('mrr', 0)) for r in rows]) * 100
        mean_top5 = np.mean([float(r.get('top5_prec', 0)) for r in rows]) * 100
        mean_top10 = np.mean([float(r.get('top10_prec', 0)) for r in rows]) * 100
        mean_dtw = np.mean([float(r.get('avg_dtw', 0)) for r in rows])
        lines.append(f"PROMEDIO GLOBAL DEL EXPERIMENTO:")
        lines.append(f"Avg LCS: {mean_lcs:.2f}% | MRR: {mean_mrr:.2f}% | Top-5: {mean_top5:.2f}% | Top-10: {mean_top10:.2f}% | DTW: {mean_dtw:.2f}")
        lines.append("=" * divider_len)

    table_content = "\n".join(lines) + "\n"
    table_path = output_dir / "comparative_table_custom_params.txt"
    try:
        table_path.write_text(table_content, encoding='utf-8')
        print(f"\n[Comparative Table] Successfully saved at {table_path}")
    except Exception as e:
        print(f"Error writing comparative table: {e}")


def run_custom_param_experiment(dataset_dir: Path, methods: list, args, base_dir: Path, cache_dir: Path):
    orig_dir = dataset_dir / args.orig_subdir
    cover_dir = dataset_dir / args.cover_subdir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = dataset_dir / output_dir
        
    if not orig_dir.exists() or not cover_dir.exists():
        print(f"Source directories '{orig_dir}' y/o '{cover_dir}' do not exist in {dataset_dir.name}. Skipping...")
        return
        
    orig_files = get_audio_files(orig_dir, match_mode=args.match_mode)
    cover_files = get_audio_files(cover_dir, match_mode=args.match_mode)
    
    if args.match_mode == "fuzzy":
        orig_files, cover_files = pair_files_fuzzy(orig_files, cover_files)
        common_ids = sorted(list(orig_files.keys()))
    else:
        common_ids = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
        
    print("\n" + "="*80)
    print(f" MC-MSA CUSTOM PARAMETER SWEEP: {dataset_dir.name} ({len(common_ids)} pairs)")
    print("="*80)
    
    if not common_ids:
        print("No valid track pairs found. Skipping...")
        return

    # Expand methods keyword
    expanded_methods = []
    for m in methods:
        if m == "tesis":
            expanded_methods.extend(['yin', 'pyin', 'melodia', 'spice', 'crepe', 'rmvpe', 'fcn_f0', 'demucs_crepe', 'demucs'])
        elif m == "all":
            expanded_methods.extend(['pyin', 'yin', 'crepe', 'rmvpe', 'spice', 'jdc', 'fcn_f0', 'melodia', 'demucs_crepe', 'demucs', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe', 'basic_pitch', 'tachibana', 'poliner', 'durrieu', 'ensemble'])
        else:
            expanded_methods.append(m)
    methods = list(dict.fromkeys(expanded_methods))

    # Parameter grid with empirical fluctuations around literature baseline values:
    # L: [4, 6, 8, 10, 12] (base L=8)
    # σ: [2, 3, 4, 5] (base σ=2, starting from 2)
    # τ_peak: [0.10, 0.15, 0.20, 0.25, 0.30] (base τ_peak=0.20)
    # y: [0.10, 0.15, 0.20, 0.25, 0.30] (base y=0.20)
    # θ_slope: [0.05, 0.10, 0.15, 0.20, 0.25] (base θ_slope=0.15)
    # τ_E: [0.15, 0.30, 0.45] (base τ_E=0.30, 0.15)
    
    grid_r = args.radii if args.radii else [4, 6, 8, 10, 12]
    grid_k = args.kernel_sizes if args.kernel_sizes else [2, 3, 4, 5]
    grid_pk = args.peak_thresholds if args.peak_thresholds else [0.10, 0.15, 0.20, 0.25, 0.30]
    grid_tl = args.tail_proportions if args.tail_proportions else [0.10, 0.15, 0.20, 0.25, 0.30]
    grid_se = args.slope_epsilons if args.slope_epsilons else [0.05, 0.10, 0.15, 0.20, 0.25]
    grid_et = args.energy_taus if args.energy_taus else [0.15, 0.30, 0.45]

    grid_combinations = list(itertools.product(grid_r, grid_k, grid_pk, grid_tl, grid_se, grid_et))
    
    # Optional subsampling / max evaluations to prevent exceedingly long runs if requested
    if getattr(args, 'max_grid_evals', None) and args.max_grid_evals < len(grid_combinations):
        step = max(1, len(grid_combinations) // args.max_grid_evals)
        grid_combinations = grid_combinations[::step][:args.max_grid_evals]

    print(f"\nEmpirical Grid Fluctuations Setup:")
    print(f"  - Checkerboard Radius (L): {grid_r}")
    print(f"  - 1D Gaussian Smoothing (σ): {grid_k}")
    print(f"  - Peak Threshold (τ_peak): {grid_pk}")
    print(f"  - Tail Portion (y): {grid_tl}")
    print(f"  - Slope Epsilon (θ_slope): {grid_se}")
    print(f"  - Energy Thresholds (τ_E): {grid_et}")
    print(f"Total parameter combinations to run per method: {len(grid_combinations)}")

    summary_path = output_dir / "mc_msa_summary_custom_params.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("method,pairs,avg_lcs,mr,mrr,mdr,map,top5_prec,top10_prec,avg_dtw,checkerboard_radius,kernel_size,peak_threshold,tail_proportion,slope_epsilon,energy_tau\n")

    for method in methods:
        print(f"\n[{method.upper()}] Loading audio features...")
        comp_cache_path = cache_dir / method / f"comparison_cache_{dataset_dir.name}.json"
        comp_cache = {}
        if comp_cache_path.exists():
            try:
                with open(comp_cache_path, 'r', encoding='utf-8') as f:
                    comp_cache = json.load(f)
            except Exception:
                pass

        # Parameter grid combinations matching thesis hyperparameters table
        grid_r = args.radii if args.radii else [4, 6, 8, 10, 12]
        grid_k = args.kernel_sizes if args.kernel_sizes else [2, 3, 4, 5, 6]
        grid_pk = args.peak_thresholds if args.peak_thresholds else [0.10, 0.15, 0.20, 0.25, 0.30]
        grid_dm = getattr(args, 'min_separations', None) or [10, 20, 30, 40, 50]
        grid_tl = args.tail_proportions if args.tail_proportions else [0.10, 0.15, 0.20, 0.25, 0.30]
        grid_se = args.slope_epsilons if args.slope_epsilons else [0.05, 0.10, 0.15, 0.20, 0.30]
        grid_et = args.energy_taus if args.energy_taus else [0.15, 0.20, 0.30, 0.40, 0.50]

        grid_combinations = list(itertools.product(grid_r, grid_k, grid_pk, grid_tl, grid_se, grid_et))
        if getattr(args, 'max_grid_evals', None) and args.max_grid_evals < len(grid_combinations):
            step = max(1, len(grid_combinations) // args.max_grid_evals)
            grid_combinations = grid_combinations[::step][:args.max_grid_evals]

        run_phase0 = args.phase in ["0", "both", "all_phases"]
        run_phase1 = args.phase in ["1", "both", "all_phases"]
        run_phase2 = args.phase in ["2", "both", "all_phases"]
        total_pairs = len(common_ids)
        analyzer = None

        # -------------------------------------------------------------
        # PHASE 0: Direct F0/Melody Extraction from Audio
        # -------------------------------------------------------------
        if run_phase0:
            print(f"\n  [PHASE 0/2] Extracting F0/Melody contours ({method.upper()}) from audio files...")
            from src.melody_analysis_v2.features import clear_deep_learning_caches
            import subprocess
            clear_deep_learning_caches()

            for pair_idx, uid in enumerate(common_ids, 1):
                # Check if JSON cache files exist first
                orig_json = cache_dir / method / f"{orig_files[uid].stem}.json"
                cover_json = cache_dir / method / f"{cover_files[uid].stem}.json"

                for f_path, j_path in [(orig_files[uid], orig_json), (cover_files[uid], cover_json)]:
                    if not j_path.exists():
                        # Run extraction in an isolated Python subprocess so ONNX/C++ memory is 100% freed by OS on exit
                        py_code = (
                            f"import sys, json, os; "
                            f"from pathlib import Path; "
                            f"sys.path.insert(0, {json.dumps(str(current_dir))}); "
                            f"from src.melody_analysis_v2.pipeline import MelodyAnalyzer; "
                            f"from run_mc_msa import safe_json; "
                            f"analyzer = MelodyAnalyzer(extraction_method={json.dumps(method)}); "
                            f"res = analyzer.analyze_file({json.dumps(str(f_path))}); "
                            f"target_p = {json.dumps(str(j_path))}; "
                            f"os.makedirs(os.path.dirname(target_p), exist_ok=True); "
                            f"f = open(target_p, 'w', encoding='utf-8'); "
                            f"json.dump(res.to_dict(), f, default=safe_json); "
                            f"f.close()"
                        )
                        cmd = [sys.executable, "-c", py_code]
                        try:
                            subprocess.run(cmd, check=True)
                        except Exception as e:
                            print(f"    Error extracting F0 for {f_path.name}: {e}")

                if pair_idx % 5 == 0 or pair_idx == total_pairs:
                    print(f"    Phase 0 Progress: {pair_idx}/{total_pairs} pairs analyzed ({pair_idx/total_pairs*100:.1f}%)")
            
            clear_deep_learning_caches()

        if not (run_phase1 or run_phase2):
            print(f"\n  [PHASE 0 Completed for {method.upper()}] F0 Extraction finished and cached.")
            continue

        # -------------------------------------------------------------
        # PHASE 1: SSM Matrix Calculation and Disk Persistence (.ssm.npy)
        # -------------------------------------------------------------
        if run_phase1:
            print(f"\n  [PHASE 1/2] Pre-computing and caching SSM matrices on disk (.ssm.npy) for {total_pairs} pairs...")
            
            for pair_idx, uid in enumerate(common_ids, 1):
                try:
                    orig_ssm = cache_dir / method / f"{orig_files[uid].stem}.ssm.npy"
                    cover_ssm = cache_dir / method / f"{cover_files[uid].stem}.ssm.npy"

                    # Only load features and compute SSM if .ssm.npy does not exist
                    if not (orig_ssm.exists() and cover_ssm.exists()):
                        feat_o, _ = load_features_fast(analyzer, orig_files[uid], method, cache_dir)
                        feat_c, _ = load_features_fast(analyzer, cover_files[uid], method, cache_dir)
                        if not orig_ssm.exists():
                            load_or_compute_ssm(feat_o, orig_files[uid], method, cache_dir)
                        if not cover_ssm.exists():
                            load_or_compute_ssm(feat_c, cover_files[uid], method, cache_dir)
                        del feat_o, feat_c
                except Exception as e:
                    print(f"    Error generating SSM cache for {uid}: {e}")
                if pair_idx % 5 == 0 or pair_idx == total_pairs:
                    print(f"    Phase 1 Progress: {pair_idx}/{total_pairs} SSMs saved to disk ({pair_idx/total_pairs*100:.1f}%)")
            
            analyzer = None
            gc.collect()

        if not run_phase2:
            print(f"\n  [PHASE 1 Completed for {method.upper()}] Features and SSM matrices (.ssm.npy) saved to disk.")
            continue

        # -------------------------------------------------------------
        # PHASE 2: Experiment Evaluation (Grid Sweep using Disk SSMs with N x N Retrieval Ranking)
        # --------------------------------------------------------------------------------------
        print(f"\n  [PHASE 2/2] Running N x N retrieval experiments over {total_pairs} pairs x {len(grid_combinations)} hyperparameter combinations...")
        
        # Pre-load features and compute/load SSMs for all originals and covers
        orig_features = {}
        cover_features = {}
        orig_ssms = {}
        cover_ssms = {}

        print("  Loading features and SSMs for all tracks...")
        for pair_idx, uid in enumerate(common_ids, 1):
            feat_o, _ = load_features_fast(None, orig_files[uid], method, cache_dir)
            feat_c, _ = load_features_fast(None, cover_files[uid], method, cache_dir)
            sim_o = load_or_compute_ssm(feat_o, orig_files[uid], method, cache_dir)
            sim_c = load_or_compute_ssm(feat_c, cover_files[uid], method, cache_dir)
            
            if feat_o is not None and feat_c is not None and sim_o is not None and sim_c is not None:
                orig_features[uid] = feat_o
                cover_features[uid] = feat_c
                orig_ssms[uid] = sim_o
                cover_ssms[uid] = sim_c

        valid_uids = list(orig_features.keys())
        n_valid = len(valid_uids)
        print(f"  Successfully loaded {n_valid} track pairs for N x N matrix evaluation.")

        r_k_pk_combos = set((c[0], c[1], c[2]) for c in grid_combinations)
        segmenter_base = MelodySegmenter()

        # Cache segmentations per (r, k, pk)
        print("  Pre-computing segmentations per (r, k, pk) cluster...")
        segs_orig_cache = {}
        segs_cover_cache = {}

        for r, k, pk in r_k_pk_combos:
            seg = MelodySegmenter(checkerboard_radius=r, kernel_size=k, peak_threshold=pk, filter_type=args.filter_type)
            segs_orig_cache[(r, k, pk)] = {}
            segs_cover_cache[(r, k, pk)] = {}

            for uid in valid_uids:
                # Original segmentation
                feat_o = orig_features[uid]
                sim_o = orig_ssms[uid]
                nov_o = seg.compute_checkerboard_novelty(sim_o)
                bounds_o = seg.find_boundaries(nov_o)
                step_o = int(np.ceil(len(feat_o.times) / segmenter_base.max_ssm_frames)) if len(feat_o.times) > segmenter_base.max_ssm_frames else 1
                indices_o = [0] + [min(int(b * step_o), len(feat_o.times) - 1) for b in bounds_o] + [len(feat_o.times) - 1]
                segs_o = [MelodySegment(start_time=float(feat_o.times[s]), end_time=float(feat_o.times[e]), start_index=int(s), end_index=int(e)) 
                          for s, e in zip(indices_o[:-1], indices_o[1:]) if e > s]
                segs_orig_cache[(r, k, pk)][uid] = segs_o

                # Cover segmentation
                feat_c = cover_features[uid]
                sim_c = cover_ssms[uid]
                nov_c = seg.compute_checkerboard_novelty(sim_c)
                bounds_c = seg.find_boundaries(nov_c)
                step_c = int(np.ceil(len(feat_c.times) / segmenter_base.max_ssm_frames)) if len(feat_c.times) > segmenter_base.max_ssm_frames else 1
                indices_c = [0] + [min(int(b * step_c), len(feat_c.times) - 1) for b in bounds_c] + [len(feat_c.times) - 1]
                segs_c = [MelodySegment(start_time=float(feat_c.times[s]), end_time=float(feat_c.times[e]), start_index=int(s), end_index=int(e)) 
                          for s, e in zip(indices_c[:-1], indices_c[1:]) if e > s]
                segs_cover_cache[(r, k, pk)][uid] = segs_c

        combo_metrics = {}

        # Evaluate each hyperparameter combination with N x N pairwise ranking
        print("  Evaluating hyperparameter combinations with N x N cross-dataset ranking...")
        for combo_idx, combo in enumerate(grid_combinations, 1):
            r, k, pk, tl, se, et = combo
            clf = MelodyClassifierThesis(
                tail_proportion=tl,
                slope_epsilon=se,
                energy_tau=et
            )

            # Classify all original and cover tracks
            seq_orig_dict = {}
            seq_cover_dict = {}

            for uid in valid_uids:
                anns_o = clf.classify(orig_features[uid], segs_orig_cache[(r, k, pk)][uid])
                seq_orig_dict[uid] = [a.label for a in anns_o]

                anns_c = clf.classify(cover_features[uid], segs_cover_cache[(r, k, pk)][uid])
                seq_cover_dict[uid] = [a.label for a in anns_c]

            # Perform N x N cover retrieval evaluation
            lcs_sims = []
            ranks = []
            mrrs = []
            top5_hits = []
            top10_hits = []

            for uid_cover in valid_uids:
                sims = []
                sim_dict = {}
                cover_seq = seq_cover_dict[uid_cover]
                
                for uid_orig in valid_uids:
                    orig_seq = seq_orig_dict[uid_orig]
                    sim = calculate_lcs(orig_seq, cover_seq)
                    sims.append((sim, uid_orig))
                    sim_dict[uid_orig] = sim

                # Correct pair self-LCS score
                correct_sim = sim_dict[uid_cover]
                lcs_sims.append(correct_sim)

                # Rank candidates descending by similarity
                sims.sort(key=lambda x: x[0], reverse=True)
                rank = -1
                for idx_rank, (sim, cand_uid) in enumerate(sims, 1):
                    if cand_uid == uid_cover:
                        rank = idx_rank
                        break

                if rank != -1:
                    ranks.append(rank)
                    mrrs.append(1.0 / rank)
                    top5_hits.append(1 if rank <= 5 else 0)
                    top10_hits.append(1 if rank <= 10 else 0)

            avg_lcs = float(np.mean(lcs_sims)) if lcs_sims else 0.0
            avg_mr = float(np.mean(ranks)) if ranks else 0.0
            avg_mrr = float(np.mean(mrrs)) if mrrs else 0.0
            avg_top5 = float(np.mean(top5_hits)) if top5_hits else 0.0
            avg_top10 = float(np.mean(top10_hits)) if top10_hits else 0.0

            combo_metrics[combo] = {
                "lcs": avg_lcs,
                "mr": avg_mr,
                "mrr": avg_mrr,
                "top5": avg_top5,
                "top10": avg_top10
            }

            if combo_idx % 10 == 0 or combo_idx == len(grid_combinations):
                print(f"    Evaluated {combo_idx}/{len(grid_combinations)} combinations -> Latest (L={r}, σ={k}, τ_pk={pk:.2f}): LCS={avg_lcs:.4f}, MRR={avg_mrr:.4f}, Top5={avg_top5*100:.2f}%")

        # Write Phase 2 results into organized CSV files
        all_combos_results = []
        summary_path = output_dir / "summary_parametros_empiricos.csv"
        detailed_csv_path = output_dir / f"experimentos_parametros_empiricos_{method}_{dataset_dir.name}.csv"
        detailed_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(detailed_csv_path, 'w', encoding='utf-8') as f_det:
            f_det.write("combo_id,method,dataset,pairs,checkerboard_radius_L,kernel_sigma,peak_threshold,tail_proportion_y,slope_epsilon,energy_tau,avg_lcs,mr,mrr,top5_prec,top10_prec\n")
            for combo_idx, (combo, m) in enumerate(combo_metrics.items(), 1):
                r, k, pk, tl, se, et = combo
                avg_lcs = m["lcs"]
                avg_mr = m["mr"]
                avg_mrr = m["mrr"]
                top5_prec = m["top5"]
                top10_prec = m["top10"]

                with open(summary_path, 'a', encoding='utf-8') as f_sum:
                    f_sum.write(f"{method},{n_valid},{avg_lcs:.4f},{avg_mr:.2f},{avg_mrr:.4f},{avg_mr:.2f},{avg_mrr:.4f},{top5_prec:.4f},{top10_prec:.4f},0.00,{r},{k},{pk:.2f},{tl:.2f},{se:.2f},{et:.2f}\n")
                
                f_det.write(f"{combo_idx},{method},{dataset_dir.name},{n_valid},{r},{k},{pk:.2f},{tl:.2f},{se:.2f},{et:.2f},{avg_lcs:.4f},{avg_mr:.2f},{avg_mrr:.4f},{top5_prec:.4f},{top10_prec:.4f}\n")
                all_combos_results.append((avg_lcs, avg_mr, avg_mrr, top5_prec, top10_prec))

        mean_lcs = float(np.mean([res[0] for res in all_combos_results])) if all_combos_results else 0.0
        mean_mr = float(np.mean([res[1] for res in all_combos_results])) if all_combos_results else 0.0
        mean_mrr = float(np.mean([res[2] for res in all_combos_results])) if all_combos_results else 0.0
        mean_top5 = float(np.mean([res[3] for res in all_combos_results])) if all_combos_results else 0.0
        mean_top10 = float(np.mean([res[4] for res in all_combos_results])) if all_combos_results else 0.0

        print(f"\n  [{method.upper()}] Experiments finished. Results saved to CSV: {detailed_csv_path}")
        print(f"  [{method.upper()}] Overall Average -> LCS: {mean_lcs:.4f}, MR: {mean_mr:.2f}, MRR: {mean_mrr:.4f}, Top-5: {mean_top5*100:.2f}%, Top-10: {mean_top10*100:.2f}%")

    print("\nCustom Parameter Experiments Completed Successfully!")


def main():
    available_methods = [
        'tesis', 'all',
        'pyin', 'yin', 'crepe', 'rmvpe', 'spice', 'jdc', 'fcn_f0',
        'melodia', 'tachibana', 'poliner', 'durrieu', 'basic_pitch',
        'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
        'bs_roformer', 'demucs', 'ensemble'
    ]
    parser = argparse.ArgumentParser(description="MC-MSA Empirical Parameters Experiment Runner.")
    parser.add_argument("--method", type=str, default="tesis",
                        choices=available_methods,
                        help="Extraction method or group to evaluate")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Base directory of the dataset")
    parser.add_argument("--orig_subdir", type=str, default="originales",
                        help="Subdirectory of original songs")
    parser.add_argument("--cover_subdir", type=str, default="covers",
                        help="Subdirectory of cover songs")
    parser.add_argument("--output_dir", type=str, default="resultados_parametros_empiricos",
                        help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory for JSON analysis cache")
    parser.add_argument("--match_mode", type=str, default="fuzzy",
                        choices=["id", "stem", "fuzzy"],
                        help="Match method")
    
    # Empirical Parameter Ranges/Fluctuations
    parser.add_argument("--radii", nargs="+", type=int, default=[4, 6, 8, 10, 12],
                        help="L: Radii range of the 2D Gaussian checkerboard kernel")
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[2, 3, 4, 5],
                        help="σ: Standard deviation range for 1D Gaussian smoothing (starts at 2)")
    parser.add_argument("--peak_thresholds", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30],
                        help="τpeak: Minimum peak height threshold range (0 to 1)")
    parser.add_argument("--tail_proportions", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30],
                        help="y: Tail proportion of non-extreme notes (starts at 0.10)")
    parser.add_argument("--slope_epsilons", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25],
                        help="εslope: Slope threshold for flat contour classification")
    parser.add_argument("--energy_taus", nargs="+", type=float, default=[0.15, 0.30, 0.45],
                        help="τE: List of energy thresholds to evaluate")
    parser.add_argument("--phase", type=str, default="both",
                        choices=["0", "1", "2", "both", "all_phases"],
                        help="Execution Phase: '0' (F0/Audio extraction only), '1' (SSM calculation to cache), '2' (Experiments only), 'both'/'all_phases' (All phases)")
    parser.add_argument("--max_grid_evals", type=int, default=150,
                        help="Maximum grid evaluation samples per method to limit execution time")
    parser.add_argument("--filter_type", type=str, choices=["gaussian", "median", "hybrid"], default="gaussian",
                        help="Type of smoothing filter on Novelty curve: 'gaussian', 'median', or 'hybrid'")
    
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()

    if args.dataset_dir is None:
        datasets = find_available_datasets(base_dir)
        if not datasets:
            manual = input("Please enter the path or name of the dataset to use: ").strip()
            args.dataset_dir = manual
        else:
            print("\n=== Dataset Selection (MC-MSA Empirical Parameters) ===")
            for i, d in enumerate(datasets, 1):
                print(f"{i}. {d}")
            print(f"{len(datasets) + 1}. [Process ALL datasets]")
            
            while True:
                choice = input(f"\nSelect a dataset (1-{len(datasets) + 1}): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(datasets):
                        args.dataset_dir = datasets[idx]
                        break
                    elif idx == len(datasets):
                        args.dataset_dir = "all"
                        break
                except ValueError:
                    if choice in datasets:
                        args.dataset_dir = choice
                        break

            # Interactive prompt to select Phase
            print("\n=== Execution Phase Selection ===")
            print("0. Phase 0 only: Pure F0/Melody Extraction from Audio -> JSON Cache")
            print("1. Phase 1 only: SSM Matrix Pre-calculation -> .ssm.npy Cache")
            print("2. Phase 2 only: Experiments and Grid Search (using existing Cache)")
            print("3. Phases 0, 1, and 2 (Full consecutive execution)")
            
            phase_choice = input("\nSelect Phase (0-3) [Default 3]: ").strip()
            if phase_choice == "0":
                args.phase = "0"
            elif phase_choice == "1":
                args.phase = "1"
            elif phase_choice == "2":
                args.phase = "2"
            else:
                args.phase = "all_phases"

            # Interactive prompt to select Method
            print("\n=== Extraction Method Selection ===")
            print("1. thesis (Main evaluation group: YIN, pYIN, Melodia, CREPE, RMVPE, Demucs, etc.)")
            print("2. all (All 18 available extraction methods)")
            print("3. Specify an individual method from list")
            
            m_choice = input("\nSelect option (1-3) [Default 1]: ").strip().lower()
            if m_choice == "2" or m_choice == "all":
                args.method = "all"
            elif m_choice == "3":
                f0_methods = ['pyin', 'yin', 'crepe', 'ensemble', 'rmvpe', 'spice', 'jdc', 'fcn_f0']
                melody_methods = [
                    'poliner', 'durrieu', 'tachibana', 'melodia', 'basic_pitch',
                    'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
                    'bs_roformer', 'demucs'
                ]
                idx_map = {}
                curr_idx = 1
                
                print("\n--- F0 Extractors (Fundamental Frequency) ---")
                for m in f0_methods:
                    print(f"  {curr_idx:2d}. {m}")
                    idx_map[curr_idx] = m
                    curr_idx += 1

                print("\n--- Melody Extractors ---")
                for m in melody_methods:
                    print(f"  {curr_idx:2d}. {m}")
                    idx_map[curr_idx] = m
                    curr_idx += 1
                
                while True:
                    sel = input(f"\nSelect a method (1-{curr_idx-1}) or type name: ").strip().lower()
                    if sel.isdigit():
                        num = int(sel)
                        if num in idx_map:
                            args.method = idx_map[num]
                            break
                        else:
                            print(f"Invalid option. Please enter a number between 1 and {curr_idx-1}.")
                    elif sel in available_methods:
                        args.method = sel
                        break
                    else:
                        print(f"Method '{sel}' not recognized. Try again.")
            else:
                args.method = "tesis"

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir

    if args.dataset_dir == "all":
        datasets = find_available_datasets(base_dir)
        for d in datasets:
            d_path = base_dir / d
            run_custom_param_experiment(d_path, [args.method], args, base_dir, cache_dir)
    else:
        d_path = base_dir / args.dataset_dir
        if not d_path.is_absolute():
            d_path = base_dir / args.dataset_dir
        run_custom_param_experiment(d_path, [args.method], args, base_dir, cache_dir)


if __name__ == "__main__":
    main()
