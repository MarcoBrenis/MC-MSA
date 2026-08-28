"""
MC-MSA CIARP 2026 Paper Benchmark Runner
=========================================
Runs the Melody-Centered Music Structure Analysis (MC-MSA) pipeline strictly according
to the specifications, parameters, and algorithms published in the CIARP 2026 paper
(formerly CLEI 2026: "MC-MSA: Melody-Centered Music Structure Analysis Approach").

Paper Reference Specifications:
-------------------------------
- Formal Function Classifier: Algorithm 1 (Antecedent 'A' vs Consequent 'C')
- Voicing Threshold (tau_voicing): 0.5
- Boundary Analysis Window (delta): 200 ms
- Pitch Slope Threshold (theta_slope): -0.15
- Energy Cutoff Threshold (theta_energy): 0.15
- Evaluated Pitch Extractors (8 methods):
    1. pYIN (Probabilistic YIN)
    2. Melodia (Essentia Predominant)
    3. SPICE (Self-supervised Pitch Estimation)
    4. CREPE (Deep Convolutional Representation)
    5. RMVPE (Robust Model for Vocal Pitch Estimation)
    6. FCN-f0 (Fully Convolutional Network)
    7. pYIN + CREPE (Hybrid Ensemble)
    8. Demucs + CREPE (Source Separation + CREPE)
- Evaluation Metrics:
    - Avg NLCS (%)
    - MRR (%)
    - Top-5 Precision (%)
    - DTW Alignment Distance
"""

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import librosa
from fastdtw import fastdtw

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodyClassifierCIARP,
    MelodySegmenter,
    MelodyFeatures,
    MelodySegmentAnnotation,
)
from src.melody_analysis_v2.classifier_thesis import calculate_lcs

# CIARP 2026 Paper Calibrated Hyperparameters (Table 1)
CIARP_VOICING_TAU = 0.5
CIARP_DELTA_MS = 200
CIARP_THETA_SLOPE = -0.15
CIARP_THETA_ENERGY = 0.15
CIARP_HANNING_L = 5
CIARP_WINDOW_W = 43
CIARP_ALPHA = 0.55
CIARP_NEIGHBOR_DIST = 10


# Default 8 Pitch Extractors evaluated in CIARP 2026 Paper (Table 2)
CIARP_EVAL_METHODS = [
    'pyin',
    'melodia',
    'spice',
    'crepe',
    'rmvpe',
    'fcn_f0',
    'pyin_crepe',
    'demucs_crepe'
]

METHOD_DISPLAY_NAMES = {
    'pyin': '1. pYIN',
    'melodia': '2. Melodia',
    'spice': '3. SPICE',
    'crepe': '4. CREPE',
    'rmvpe': '5. RMVPE',
    'fcn_f0': '6. FCN-f0',
    'pyin_crepe': '7. pYIN + CREPE',
    'demucs_crepe': '8. Demucs + CREPE',
}


def get_audio_files(directory_path: Path) -> Dict[str, Path]:
    """Finds audio files (.mp3, .wav) and maps normalized stem names to paths."""
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            name = f.stem.lower()
            name = re.sub(r'[-_](cover|originales|original|orig|ref|covers|version|var)', '', name)
            name = re.sub(r'^\d+\s*[-_]?\s*', '', name)
            result[name] = f
    return result


def compute_dtw_distance(pitch1: np.ndarray, pitch2: np.ndarray) -> float:
    """Calculates DTW alignment distance between two pitch contours."""
    valid1 = pitch1[(pitch1 > 0) & (~np.isnan(pitch1))]
    valid2 = pitch2[(pitch2 > 0) & (~np.isnan(pitch2))]
    if len(valid1) < 5 or len(valid2) < 5:
        return 999.0
    
    # Downsample if sequences are long
    max_len = 1000
    if len(valid1) > max_len:
        valid1 = valid1[::int(len(valid1)/max_len)]
    if len(valid2) > max_len:
        valid2 = valid2[::int(len(valid2)/max_len)]
        
    try:
        distance, _ = fastdtw(valid1.reshape(-1, 1), valid2.reshape(-1, 1), dist=lambda x, y: abs(x[0] - y[0]))
        return float(distance / max(len(valid1), len(valid2)))
    except Exception:
        return 999.0


def run_ciarp_benchmark(
    orig_dir: Path,
    cover_dir: Path,
    methods: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """Executes the full CIARP 2026 paper benchmark on paired original-cover tracks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orig_files = get_audio_files(orig_dir)
    cover_files = get_audio_files(cover_dir)
    
    common_keys = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
    print(f"\n========================================================")
    print(f"  MC-MSA CIARP 2026 Paper Benchmark Pipeline")
    print(f"========================================================")
    print(f"Found {len(common_keys)} matched pairs (Original - Cover).")
    print(f"Methods to evaluate: {', '.join(methods)}")
    print(f"Classifier: MelodyClassifierCIARP (Algorithm 1)")
    print(f"Hyperparameters: theta_slope={CIARP_THETA_SLOPE}, theta_energy={CIARP_THETA_ENERGY}")
    print(f"========================================================\n")

    segmenter = MelodySegmenter(
        adaptive_threshold=True,
        hanning_size=CIARP_HANNING_L,
        window_w=CIARP_WINDOW_W,
        alpha=CIARP_ALPHA,
        neighbor_dist=CIARP_NEIGHBOR_DIST
    )
    classifier = MelodyClassifierCIARP(
        slope_epsilon=abs(CIARP_THETA_SLOPE),
        energy_tau=CIARP_THETA_ENERGY
    )
    
    benchmark_results = {}
    
    for method in methods:
        print(f"\n---> Evaluating Method: {METHOD_DISPLAY_NAMES.get(method, method)}...")
        analyzer = MelodyAnalyzer(
            extraction_method=method,
            classifier=classifier,
            segmenter=segmenter
        )

        
        nlcs_scores = []
        dtw_distances = []
        reciprocal_ranks = []
        top5_hits = []
        
        pair_sequences = {}
        pair_pitches = {}
        
        for key in common_keys:
            orig_path = orig_files[key]
            cover_path = cover_files[key]
            
            try:
                res_orig = analyzer.analyze_file(str(orig_path))
                res_cover = analyzer.analyze_file(str(cover_path))
                
                seq_orig = [seg.label for seg in res_orig.segments if seg.label in ['Antecedent', 'Consequent', 'A', 'C']]
                seq_cover = [seg.label for seg in res_cover.segments if seg.label in ['Antecedent', 'Consequent', 'A', 'C']]
                
                # Normalize labels to A/C
                seq_orig = ['A' if x == 'Antecedent' else ('C' if x == 'Consequent' else x) for x in seq_orig]
                seq_cover = ['A' if x == 'Antecedent' else ('C' if x == 'Consequent' else x) for x in seq_cover]
                
                pair_sequences[key] = (seq_orig, seq_cover)
                pair_pitches[key] = (res_orig.features.pitch_midi, res_cover.features.pitch_midi)
                
                nlcs = calculate_lcs(seq_orig, seq_cover)
                dtw_dist = compute_dtw_distance(res_orig.features.pitch_midi, res_cover.features.pitch_midi)
                
                nlcs_scores.append(nlcs * 100.0)
                if dtw_dist < 990:
                    dtw_distances.append(dtw_dist)
                    
            except Exception as e:
                print(f"Error processing pair {key} with {method}: {e}")
                
        # Retrieval evaluation across database (MRR, Top-5)
        keys_list = list(pair_sequences.keys())
        for q_idx, q_key in enumerate(keys_list):
            q_seq_orig, q_seq_cover = pair_sequences[q_key]
            sims = []
            for r_key in keys_list:
                r_seq_orig, r_seq_cover = pair_sequences[r_key]
                sim = calculate_lcs(q_seq_cover, r_seq_orig)
                sims.append((r_key, sim))
            
            # Sort by similarity descending
            sims.sort(key=lambda x: x[1], reverse=True)
            ranked_keys = [x[0] for x in sims]
            
            try:
                rank = ranked_keys.index(q_key) + 1
            except ValueError:
                rank = len(keys_list)
                
            reciprocal_ranks.append(1.0 / rank)
            top5_hits.append(1.0 if rank <= 5 else 0.0)
            
        avg_nlcs = float(np.mean(nlcs_scores)) if nlcs_scores else 0.0
        avg_mrr = float(np.mean(reciprocal_ranks)) * 100.0 if reciprocal_ranks else 0.0
        avg_top5 = float(np.mean(top5_hits)) * 100.0 if top5_hits else 0.0
        avg_dtw = float(np.mean(dtw_distances)) if dtw_distances else 0.0
        
        benchmark_results[method] = {
            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "nlcs_percent": avg_nlcs,
            "mrr_percent": avg_mrr,
            "top5_percent": avg_top5,
            "dtw_distance": avg_dtw,
            "num_pairs": len(nlcs_scores)
        }
        
        print(f"   Avg NLCS: {avg_nlcs:.2f}% | MRR: {avg_mrr:.2f}% | Top-5: {avg_top5:.2f}% | DTW: {avg_dtw:.2f}")

    # Generate Summary Latex Table (Matching Table 2 of CIARP 2026 Paper)
    latex_table_path = output_dir / "ciarp_table2_results.tex"
    with open(latex_table_path, "w") as f:
        f.write("% CIARP 2026 Paper Table 2: CSI Performance Across Melodic Extraction Methods\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{CSI Performance Across Melodic Extraction Methods}\n")
        f.write("\\label{tab:ciarp_results}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Method & NLCS (\\%) & MRR (\\%) & Top-5 (\\%) & DTW \\\\\n\\hline\n")
        for m, res in benchmark_results.items():
            f.write(f"{res['display_name']} & {res['nlcs_percent']:.2f} & {res['mrr_percent']:.2f} & {res['top5_percent']:.2f} & {res['dtw_distance']:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Save JSON summary report
    json_report_path = output_dir / "ciarp_benchmark_summary.json"
    with open(json_report_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n========================================================")
    print(f" Benchmark Complete!")
    print(f" Summary LaTeX Table saved to: {latex_table_path}")
    print(f" Summary JSON Report saved to: {json_report_path}")
    print(f"========================================================\n")
    
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="MC-MSA CIARP 2026 Paper Benchmark Runner")
    parser.add_argument("--orig-dir", type=str, default="dataset_Acad/Originales", help="Path to originals folder")
    parser.add_argument("--cover-dir", type=str, default="dataset_Acad/Covers", help="Path to covers folder")
    parser.add_argument("--output-dir", type=str, default="outputs_ciarp", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=CIARP_EVAL_METHODS, help="Extraction methods to evaluate")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    orig_path = (script_dir / args.orig_dir).resolve()
    cover_path = (script_dir / args.cover_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    
    run_ciarp_benchmark(orig_path, cover_path, args.methods, output_path)


if __name__ == "__main__":
    main()
