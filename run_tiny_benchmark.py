import os
import re
import json
import argparse
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from src.melody_analysis_v2 import MelodyAnalyzer, MelodyClassifierPaper, MelodyFeatures, MelodySegmentAnnotation
from src.melody_analysis_v2.classifier_paper import calculate_lcs
from src.melody_analysis_v2.segmenter import MelodySegment
from src.melody_analysis_v2.pipeline import MelodyAnalysisResult

# Reutilizamos las funciones de ploteo de run_benchmark
from run_benchmark import (
    plot_caplin_bands, 
    plot_caplin_contour,
    plot_boundary_detection,
    plot_self_similarity,
    plot_spectrogram_with_segments
)

def load_or_analyze(analyzer, file_path, method, cache_dir):
    cache_path = cache_dir / method / f"{file_path.stem}.json"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            data = json.load(f)
        features = MelodyFeatures(
            times=np.array(data["times"]),
            pitch_midi=np.array(data["pitch_midi"]),
            confidence=np.array(data["confidence"]),
            energy=np.array(data["energy"])
        )
        segments = [
            MelodySegmentAnnotation(
                segment=MelodySegment(s["start_time"], s["end_time"], 0, 0),
                label=s["label"],
                confidence=s["confidence"],
                descriptor=s["descriptor"]
            )
            for s in data["segments"]
        ]
        for s in segments:
            s.segment.start_index = int(np.searchsorted(features.times, s.segment.start_time))
            s.segment.end_index = int(np.searchsorted(features.times, s.segment.end_time))
        return MelodyAnalysisResult(features=features, segments=segments)
    
    result = analyzer.analyze_file(str(file_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(result.to_dict(), f)
    return result

def main():
    available_methods = ['all', 'pyin', 'melodia', 'crepe', 'ensemble', 'demucs_crepe', 'bs_roformer_rmvpe', 'rmvpe']
    parser = argparse.ArgumentParser(description="Tiny Benchmark con soporte de método y caché.")
    parser.add_argument("--method", type=str, default=None, 
                        choices=available_methods,
                        help="Método de extracción a utilizar")
    args = parser.parse_args()

    if args.method is None:
        print("\n=== Selección de Método de Extracción ===")
        for i, m in enumerate(available_methods, 1):
            print(f"{i}. {m}")
        
        while True:
            try:
                choice = input(f"\nSeleccione un método (1-{len(available_methods)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(available_methods):
                    args.method = available_methods[idx]
                    break
                else:
                    print(f"Error: Por favor seleccione un número entre 1 y {len(available_methods)}.")
            except ValueError:
                # Si el usuario escribe el nombre del método directamente, también lo aceptamos
                if choice.strip().lower() in available_methods:
                    args.method = choice.strip().lower()
                    break
                print("Error: Entrada no válida. Ingrese el número del método o el nombre.")

    base_dir = Path(__file__).parent.absolute()
    dataset_dir = base_dir / "dataset_clei"
    orig_dir = dataset_dir / "originales"
    cover_dir = dataset_dir / "covers"
    cache_dir = base_dir / "cache_tiny"
    
    orig_files = {
        2: next(orig_dir.glob("02 *.mp3")),
        4: next(orig_dir.glob("04 *.mp3"))
    }
    cover_files = {
        2: next(cover_dir.glob("02 *.mp3")),
        4: next(cover_dir.glob("04 *.mp3"))
    }
    common_ids = [2, 4]
    
    if args.method == 'all':
        methods = ['crepe', 'bs_roformer_rmvpe']
    else:
        methods = [args.method]

    classifier = MelodyClassifierPaper()
    print(f"\nIniciando TINY BENCHMARK | Métodos: {methods}")
    
    for method in methods:
        print(f"\n[{method}] Analizando...")
        analyzer = MelodyAnalyzer(extraction_method=method, classifier=classifier)
        
        res_originals = {}
        total_p = len(common_ids)
        for i, uid in enumerate(common_ids, 1):
            print(f"  [{i}/{total_p}] ({i/total_p:.1%}) Cargando original: ID {uid}...", end='\r')
            res_originals[uid] = load_or_analyze(analyzer, orig_files[uid], method, cache_dir)
        print(f"\n  Originales cargados.")
        
        lcs_correct_list, dtw_correct_list, mrr_sum, top5_hits, valid_covers = [], [], 0.0, 0, 0
        best_lcs, best_uid, best_pair_res = -1.0, None, None
        detailed_results = []
        
        for i, uid_cover in enumerate(common_ids, 1):
            print(f"  [{i}/{total_p}] ({i/total_p:.1%}) Cargando cover: ID {uid_cover}...", end='\r')
            res_cover = load_or_analyze(analyzer, cover_files[uid_cover], method, cache_dir)
            seq_cover = [s.label for s in res_cover.segments]
            
            pitch_midi_c = res_cover.features.pitch_midi
            f0_cover = np.nan_to_num(np.where(pitch_midi_c > 0, 440.0 * np.power(2.0, (pitch_midi_c - 69.0) / 12.0), 0))
            
            similarities = []
            for uid_orig in common_ids:
                seq_orig = [s.label for s in res_originals[uid_orig].segments]
                sim = calculate_lcs(seq_orig, seq_cover)
                similarities.append((sim, uid_orig))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            rank, true_sim = -1, 0.0
            for idx, (sim, r_uid) in enumerate(similarities):
                if r_uid == uid_cover:
                    rank, true_sim = idx + 1, sim
                    break
                    
            if rank != -1:
                valid_covers += 1
                mrr_sum += 1.0 / rank
                if rank <= 5: top5_hits += 1
                lcs_correct_list.append(true_sim)
                
                if true_sim > best_lcs:
                    best_lcs = true_sim
                    best_uid = uid_cover
                    best_pair_res = (res_originals[uid_cover], res_cover)
                    print(f"\n  [Mejor Match] ID {uid_cover} con LCS={best_lcs:.4f}. Generando gráficas completas...")
                    out_dir = base_dir / "salidas_tiny_benchmark" / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    plot_caplin_bands(best_pair_res[0], best_pair_res[1], out_dir / "fig_qualitative_bands.pdf")
                    plot_caplin_contour(best_pair_res[0], best_pair_res[1], out_dir / "fig_qualitative_contour.pdf")
                    
                    # Advanced Plots
                    try:
                        # Re-analyze to ensure we have novelty/ssm (not in cache)
                        print(f"  Re-analizando best pair para obtener matrices de visualización...", end='\r')
                        res_orig_full = analyzer.analyze_file(str(orig_files[uid_cover]))
                        res_cover_full = analyzer.analyze_file(str(cover_files[uid_cover]))
                        
                        plot_boundary_detection(res_orig_full, output_path=out_dir / "fig_novelty_orig.pdf")
                        plot_boundary_detection(res_cover_full, output_path=out_dir / "fig_novelty_cover.pdf")
                        
                        if res_orig_full.self_similarity is not None:
                            plot_self_similarity(res_orig_full, output_path=out_dir / "fig_ssm_orig.pdf")
                        if res_cover_full.self_similarity is not None:
                            plot_self_similarity(res_cover_full, output_path=out_dir / "fig_ssm_cover.pdf")
                            
                        # Spectrograms
                        for name, res_f, path_audio in [("orig", res_orig_full, orig_files[uid_cover]), 
                                                     ("cover", res_cover_full, cover_files[uid_cover])]:
                            audio = res_f.normalized_audio
                            sr = res_f.sample_rate
                            if audio is None or sr is None:
                                audio, sr = librosa.load(path_audio, sr=22050)
                            plot_spectrogram_with_segments(audio, sr, res_f, output_path=out_dir / f"fig_spectrogram_{name}.pdf")
                        print(f"  Gráficas completas generadas exitosamente.{" "*20}")
                    except Exception as plot_err:
                        print(f"\n  Advertencia al generar gráficas avanzadas: {plot_err}")
                
                dtw_val = 0.0
                pitch_o = res_originals[uid_cover].features.pitch_midi
                f0_orig = np.nan_to_num(np.where(pitch_o > 0, 440.0 * np.power(2.0, (pitch_o - 69.0) / 12.0), 0))
                try:
                    D, wp = librosa.sequence.dtw(f0_cover.reshape(1, -1), f0_orig.reshape(1, -1))
                    dtw_val = D[-1, -1] / len(wp)
                    dtw_correct_list.append(dtw_val)
                except: pass
                
                detailed_results.append(f"ID {uid_cover:02d} | LCS: {true_sim:.4f} | Rank: {rank:2d} | DTW: {dtw_val:.4f}")
                    
        # Stats
        avg_lcs = np.mean(lcs_correct_list) if lcs_correct_list else 0.0
        mrr = mrr_sum / valid_covers if valid_covers > 0 else 0.0
        top5_prec = top5_hits / valid_covers if valid_covers > 0 else 0.0
        avg_dtw = np.mean(dtw_correct_list) if dtw_correct_list else 0.0
        
        print(f"\n[{method}] Resumen | LCS: {avg_lcs:.4f} | MRR: {mrr:.4f} | Top5: {top5_prec:.1%} | DTW: {avg_dtw:.4f}")
        
        # Reporte TXT Detallado
        out_method_dir = base_dir / "salidas_tiny_benchmark" / method
        out_method_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_method_dir / "reporte_detallado.txt"
        with open(report_path, 'w') as f:
            f.write(f"REPORTE DETALLADO - TINY BENCHMARK - METODO: {method}\n")
            f.write("="*50 + "\n")
            if best_uid is not None:
                f.write(f"IMAGENES GENERADAS PARA EL MEJOR MATCH (LCS = {best_lcs:.4f}):\n")
                f.write(f"  ID: {best_uid}\n")
                f.write(f"  Original: {orig_files[best_uid].name}\n")
                f.write(f"  Cover:    {cover_files[best_uid].name}\n")
                f.write("="*50 + "\n")
            f.write("\n".join(detailed_results) + "\n")
            f.write("="*50 + "\n")
            f.write(f"RESUMEN:\n")
            f.write(f"LCS Promedio: {avg_lcs:.4f}\n")
            f.write(f"MRR:          {mrr:.4f}\n")
            f.write(f"Top-5 Prec.:  {top5_prec:.2%}\n")
            f.write(f"DTW Promedio: {avg_dtw:.4f}\n")
        
        print(f"\n[{method}] Finalizado.")



if __name__ == "__main__":
    main()
