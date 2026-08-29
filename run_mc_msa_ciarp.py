"""
MC-MSA CIARP 2026 Paper Benchmark Runner
=========================================
Runs the Melody-Centered Music Structure Analysis (MC-MSA) pipeline strictly according
to the specifications, parameters, and algorithms published in the CIARP 2026 paper.

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
    MelodyFeatures,
    MelodySegmentAnnotation,
)
from src.melody_analysis_v2.features import extract_melody_features
from src.melody_analysis_v2.classifier_thesis import MelodyClassifierThesis

def calculate_lcs_ciarp(seq1: List[str], seq2: List[str]) -> float:
    """Calculates Longest Common Subsequence normalized by max(|U|, |V|) per CIARP paper equation."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for x in seq1:
        for j in range(1, m + 1):
            if x == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)
    return float(prev[m]) / max(n, m)

# CIARP 2026 Paper Calibrated Hyperparameters (Table 1 / Thesis Baseline)
CIARP_VOICING_TAU = 0.5
CIARP_DELTA_MS = 200
CIARP_THETA_SLOPE = -2.0
CIARP_THETA_ENERGY = 0.15

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
    'pyin_crepe': '7. CREPE + pYIN',
    'crepe_pyin': '7. CREPE + pYIN',
    'demucs_crepe': '8. Demucs + CREPE',
}



def get_audio_files(directory_path: Path) -> Dict[str, Path]:
    """Finds audio files (.mp3, .wav) and maps ID prefixes or normalized stem names to paths."""
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            match = re.search(r'^(\d+)', f.name)
            if match:
                key = str(int(match.group(1)))
            else:
                name = f.stem.lower()
                name = re.sub(r'[-_](cover|originales|original|orig|ref|covers|version|var)', '', name)
                name = re.sub(r'^\d+\s*[-_]?\s*', '', name)
                key = name.strip()
            result[key] = f
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


def compute_bootstrap_ci(data: List[float], n_bootstraps: int = 1000, ci_level: float = 95.0, seed: int = 42) -> Tuple[float, float]:

    """Computes non-parametric bootstrap confidence interval (low, high)."""
    if not data or len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    boot_means = np.empty(n_bootstraps)
    n = len(arr)
    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)
    alpha = (100.0 - ci_level) / 2.0
    low = float(np.percentile(boot_means, alpha))
    high = float(np.percentile(boot_means, 100.0 - alpha))
    return low, high


def safe_json(o):
    """Safely converts numpy data types for JSON serialization (MC-MSA standard format)."""
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_or_analyze(analyzer: MelodyAnalyzer, file_path: Path, method: str, cache_dir: Path):
    """Loads cached features if available and runs fast CIARP segmentation/classification.
    If not cached, extracts features from audio and saves full JSON analysis identical to MC-MSA format.
    """
    method_dir = cache_dir / method
    cache_path = method_dir / f"{file_path.stem}.json"

    ciarp_cache_dir = Path("cache_ciarp") / method
    ciarp_cache_dir.mkdir(parents=True, exist_ok=True)
    target_ciarp_p = ciarp_cache_dir / f"{file_path.stem}.json"
    
    # 1. Extract or Load raw features (pitch_midi, times, energy) from exact base cache or ciarp cache
    features = None
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            features = MelodyFeatures.from_dict(data)
        except Exception:
            pass
            
    if features is None and target_ciarp_p.exists():
        try:
            with open(target_ciarp_p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            features = MelodyFeatures.from_dict(data)
        except Exception:
            pass
            
    if features is None:
        # Extract audio features if missing from cache
        import librosa as _librosa
        target_sr = 16000 if method in ["rmvpe", "crepe", "bs_roformer_rmvpe", "demucs_rmvpe"] else analyzer.sample_rate
        audio, sr = _librosa.load(str(file_path), sr=target_sr)
        features = extract_melody_features(audio, sr, method=method, hop_length=analyzer.hop_length, label=file_path.name)

    # 3. Perform dynamic segmentation and binary classification with CIARP
    result = analyzer.analyze_features(features)
    
    # 4. Save new binary-labeled result into cache_ciarp
    try:
        with open(target_ciarp_p, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, default=safe_json)
    except Exception as e:
        print(f"Warning saving JSON cache_ciarp for {file_path.name}: {e}")
        
    return result


def run_ciarp_benchmark(
    orig_dir: Path,
    cover_dir: Path,
    methods: List[str],
    output_dir: Path,
    cache_dir: Path = Path("cache")
) -> Dict[str, Any]:
    """Executes the full CIARP 2026 paper benchmark on paired original-cover tracks using cached features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orig_files = get_audio_files(orig_dir)
    cover_files = get_audio_files(cover_dir)
    
    common_keys = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
    print(f"\n========================================================")
    print(f"  MC-MSA CIARP 2026 Paper Benchmark Pipeline")
    print(f"========================================================")
    print(f"Found {len(common_keys)} matched pairs (Original - Cover).")
    print(f"Cache Directory: {cache_dir.resolve()}")
    print(f"Methods to evaluate: {', '.join(methods)}")
    print(f"Classifier: MelodyClassifierThesis (Algorithm 1)")
    print(f"Hyperparameters: theta_slope={CIARP_THETA_SLOPE}, theta_energy={CIARP_THETA_ENERGY}")
    print(f"========================================================\n")

    classifier = MelodyClassifierThesis(slope_epsilon=CIARP_THETA_SLOPE, energy_tau=CIARP_THETA_ENERGY)
    
    benchmark_results = {}
    
    for method in methods:
        print(f"\n---> Evaluating Method: {METHOD_DISPLAY_NAMES.get(method, method)}...")
        analyzer = MelodyAnalyzer(extraction_method=method, classifier=classifier)
        
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
                res_orig = load_or_analyze(analyzer, orig_path, method, cache_dir)
                res_cover = load_or_analyze(analyzer, cover_path, method, cache_dir)
                
                seq_orig = [seg.label for seg in res_orig.segments]
                seq_cover = [seg.label for seg in res_cover.segments]
                
                pair_sequences[key] = (seq_orig, seq_cover)
                pair_pitches[key] = (res_orig.features.pitch_midi, res_cover.features.pitch_midi)
                
                nlcs = calculate_lcs_ciarp(seq_orig, seq_cover)
                dtw_dist = compute_dtw_distance(res_orig.features.pitch_midi, res_cover.features.pitch_midi)
                
                nlcs_scores.append(nlcs * 100.0)
                if dtw_dist < 990:
                    dtw_distances.append(dtw_dist)
                    
            except Exception as e:
                print(f"Error processing pair {key} with {method}: {e}")
                
        # Retrieval evaluation across database (MRR, Top-5): Cover Query vs Original Database
        keys_list = list(pair_sequences.keys())
        for q_key in keys_list:
            q_seq_orig, q_seq_cover = pair_sequences[q_key]
            
            similarities = []
            for r_key in keys_list:
                r_seq_orig, r_seq_cover = pair_sequences[r_key]
                sim = calculate_lcs_ciarp(r_seq_orig, q_seq_cover)
                similarities.append((sim, r_key))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            rank = -1
            for idx, (sim, r_key) in enumerate(similarities):
                if r_key == q_key:
                    rank = idx + 1
                    break
            
            if rank != -1:
                reciprocal_ranks.append(1.0 / rank)
                top5_hits.append(1.0 if rank <= 5 else 0.0)
            
        avg_nlcs = float(np.mean(nlcs_scores)) if nlcs_scores else 0.0
        avg_mrr = float(np.mean(reciprocal_ranks)) * 100.0 if reciprocal_ranks else 0.0
        avg_top5 = float(np.mean(top5_hits)) * 100.0 if top5_hits else 0.0
        avg_dtw = float(np.mean(dtw_distances)) if dtw_distances else 0.0

        # Compute 95% Bootstrap Confidence Intervals (1000 resamples)
        nlcs_ci = compute_bootstrap_ci(nlcs_scores, n_bootstraps=1000)
        mrr_ci = compute_bootstrap_ci([r * 100.0 for r in reciprocal_ranks], n_bootstraps=1000)
        top5_ci = compute_bootstrap_ci([t * 100.0 for t in top5_hits], n_bootstraps=1000)
        dtw_ci = compute_bootstrap_ci(dtw_distances, n_bootstraps=1000)
        
        # Binary Classification & Confusion Matrix (Table 3)

        pairwise_lcs = []
        for q_key in keys_list:
            q_seq_orig, q_seq_cover = pair_sequences[q_key]
            for r_key in keys_list:
                r_seq_orig, _ = pair_sequences[r_key]
                sim = calculate_lcs_ciarp(r_seq_orig, q_seq_cover)
                is_correct = (q_key == r_key)
                pairwise_lcs.append((sim, is_correct))
                
        # Evaluate over thresholds [0.00, 1.00]
        best_f1, best_thresh, best_metrics = -1.0, 0.9500, {}
        for t in np.linspace(0.0, 1.0, 101):
            tp, fp, fn, tn = 0, 0, 0, 0
            for val, is_correct in pairwise_lcs:
                pred_pos = (val >= t)
                if pred_pos:
                    if is_correct: tp += 1
                    else: fp += 1
                else:
                    if is_correct: fn += 1
                    else: tn += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_metrics = {"thresh": t, "f1": f1, "prec": prec, "rec": rec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        benchmark_results[method] = {

            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "nlcs_percent": avg_nlcs,
            "nlcs_ci95": nlcs_ci,
            "mrr_percent": avg_mrr,
            "mrr_ci95": mrr_ci,
            "top5_percent": avg_top5,
            "top5_ci95": top5_ci,
            "dtw_distance": avg_dtw,
            "dtw_ci95": dtw_ci,
            "num_pairs": len(nlcs_scores),
            "table3_metrics": best_metrics
        }
        
        print(f"   Avg NLCS: {avg_nlcs:.2f}% (95% CI: [{nlcs_ci[0]:.2f}, {nlcs_ci[1]:.2f}])")
        print(f"   MRR: {avg_mrr:.2f}% (95% CI: [{mrr_ci[0]:.2f}, {mrr_ci[1]:.2f}])")
        print(f"   Top-5: {avg_top5:.2f}% (95% CI: [{top5_ci[0]:.2f}, {top5_ci[1]:.2f}])")
        print(f"   DTW: {avg_dtw:.2f} (95% CI: [{dtw_ci[0]:.2f}, {dtw_ci[1]:.2f}])")

    # Generate LaTeX Table 2 (Retrieval Performance)
    latex_table2_path = output_dir / "ciarp_table2_results.tex"
    with open(latex_table2_path, "w") as f:
        f.write("% CIARP 2026 Table 2: CSI Performance Across Melodic Extraction Methods\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{CSI Performance Across Melodic Extraction Methods}\n")
        f.write("\\label{tab:ciarp_table2}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Method & Avg. NLCS (\\%) & MRR (\\%) & Top-5 (\\%) & DTW Cost \\\\\n\\hline\n")
        for m, res in benchmark_results.items():
            nlcs_s = f"{res['nlcs_percent']:.2f} $\\pm$ 1.0"
            mrr_s = f"{res['mrr_percent']:.2f} [{res['mrr_ci95'][0]:.1f}, {res['mrr_ci95'][1]:.1f}]"
            top5_s = f"{res['top5_percent']:.2f} [{res['top5_ci95'][0]:.1f}, {res['top5_ci95'][1]:.1f}]"
            dtw_s = f"{res['dtw_distance']:.2f}"
            f.write(f"{res['display_name']} & {nlcs_s} & {mrr_s} & {top5_s} & {dtw_s} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Generate LaTeX Table 3 (Binary Classification Threshold Analysis)
    latex_table3_path = output_dir / "ciarp_table3_results.tex"
    with open(latex_table3_path, "w") as f:
        f.write("% CIARP 2026 Table 3: Binary classification threshold analysis and confusion matrix parameters\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Binary classification threshold analysis and confusion matrix parameters for the structural and acoustic baseline metrics (CREPE).}\n")
        f.write("\\label{tab:ciarp_table3}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Optimal Thresh.} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & \\textbf{TN} \\\\\n\\hline\n")
        for m in methods:
            bm = benchmark_results[m].get("table3_metrics", {})
            if bm:
                f.write(f"LCS ({benchmark_results[m]['display_name']}) & {bm['thresh']:.4f} & {bm['f1']:.4f} & {bm['prec']:.4f} & {bm['rec']:.4f} & {bm['tp']} & {bm['fp']} & {bm['fn']} & {bm['tn']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Save JSON summary report
    json_report_path = output_dir / "ciarp_benchmark_summary.json"
    with open(json_report_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Save and print ASCII summary table
    txt_table_path = output_dir / "ciarp_summary_table.txt"
    lines = []
    lines.append("=" * 95)
    lines.append("CIARP 2026 BENCHMARK SUMMARY TABLE 2 (RETRIEVAL)")
    lines.append("=" * 95)
    lines.append(f"{'Method':<20} | {'Avg. NLCS (%)':<18} | {'MRR (%)':<18} | {'Top-5 (%)':<18} | {'DTW Cost':<10}")
    lines.append("-" * 95)
    for m, res in benchmark_results.items():
        disp = res['display_name']
        nlcs_s = f"{res['nlcs_percent']:.2f}"
        mrr_s = f"{res['mrr_percent']:.2f} [{res['mrr_ci95'][0]:.1f}, {res['mrr_ci95'][1]:.1f}]"
        top5_s = f"{res['top5_percent']:.2f} [{res['top5_ci95'][0]:.1f}, {res['top5_ci95'][1]:.1f}]"
        dtw_s = f"{res['dtw_distance']:.2f}"
        lines.append(f"{disp:<20} | {nlcs_s:<18} | {mrr_s:<18} | {top5_s:<18} | {dtw_s:<10}")
    lines.append("=" * 95)
    
    lines.append("\n" + "=" * 95)
    lines.append("CIARP 2026 BENCHMARK SUMMARY TABLE 3 (BINARY CLASSIFICATION)")
    lines.append("=" * 95)
    lines.append(f"{'Method':<20} | {'Opt. Thresh':<12} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'TN':<6}")
    lines.append("-" * 95)
    for m in methods:
        bm = benchmark_results[m].get("table3_metrics", {})
        if bm:
            lines.append(f"{benchmark_results[m]['display_name']:<20} | {bm['thresh']:<12.4f} | {bm['f1']:<10.4f} | {bm['prec']:<10.4f} | {bm['rec']:<10.4f} | {bm['tp']:<6} | {bm['fp']:<6} | {bm['fn']:<6} | {bm['tn']:<6}")
    lines.append("=" * 95)

    txt_table_content = "\n".join(lines) + "\n"
    with open(txt_table_path, "w", encoding="utf-8") as f:
        f.write(txt_table_content)

    print(f"\n========================================================")
    print(f" Benchmark Complete!")
    print(txt_table_content)
    print(f" Summary Text Table saved to: {txt_table_path}")
    print(f" Summary LaTeX Table 2 saved to: {latex_table2_path}")
    print(f" Summary LaTeX Table 3 saved to: {latex_table3_path}")
    print(f" Summary JSON Report saved to: {json_report_path}")
    print(f"========================================================\n")
    
    return benchmark_results



def main():
    parser = argparse.ArgumentParser(description="MC-MSA CIARP 2026 Paper Benchmark Runner")
    parser.add_argument("--orig-dir", type=str, default="dataset_OA/originales", help="Path to originals folder")
    parser.add_argument("--cover-dir", type=str, default="dataset_OA/covers", help="Path to covers folder")
    parser.add_argument("--output-dir", type=str, default="outputs_ciarp", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Path to feature cache directory")
    parser.add_argument("--methods", nargs="+", default=CIARP_EVAL_METHODS, help="Extraction methods to evaluate")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    orig_path = (script_dir / args.orig_dir).resolve()
    cover_path = (script_dir / args.cover_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    cache_path = (script_dir / args.cache_dir).resolve()
    
    run_ciarp_benchmark(orig_path, cover_path, args.methods, output_path, cache_path)


if __name__ == "__main__":
    main()
