"""
MC-MSA CIARP2 Benchmark Runner with Time & Frequency Aligned DTW
=================================================================
Runs the Melody-Centered Music Structure Analysis (MC-MSA) pipeline with 
Time & Frequency Aligned DTW (Dynamic Time and Frequency Warping / Key-Invariant DTW).

Key Enhancements in CIARP2:
---------------------------
1. Time & Frequency Aligned DTW: Centered & Shift-Invariant DTW for both MIDI semitones & Hz.
2. 3-Phase Modular Architecture:
   - Phase 1: Feature Extraction (F0 & Energy -> cache_ciarp2/phase1_f0/)
   - Phase 2: Matrix & SSM Generation (Key-Invariant DTW & State Sequences -> cache_ciarp2/phase2_matrices/)
   - Phase 3: Classifier & Report Generation (LaTeX Table 2, Table 3, Summary -> outputs_ciarp2/)
3. Interactive & CLI Cache Management: Selective flushing of Phase 1, Phase 2, Phase 3, or all caches.
"""

import os
import re
import sys
import shutil
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


# CIARP 2026 Paper Calibrated Hyperparameters
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
    'demucs_crepe': '8. Demucs + CREPE',
}


def get_audio_files(directory_path: Path) -> Dict[str, Path]:
    """Scans folder and maps pair keys (numeric ID or cleaned track title) to audio file paths."""
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg']:
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


def dynamic_frequency_warping(pitch1: np.ndarray, pitch2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Implements Dynamic Frequency Warping (DFW, Matsumoto 1987 JASA).
    Aligns pitch contours non-linearly along the frequency/pitch axis (dual of DTW).
    """
    if len(pitch1) == 0 or len(pitch2) == 0:
        return pitch2, 999.0
        
    # 1. Frequency shift alignment (Median pitch centering)
    med1 = float(np.median(pitch1))
    med2 = float(np.median(pitch2))
    p2_shifted = pitch2 - (med2 - med1)
    
    # 2. Dynamic programming cost matrix along frequency axis
    N, M = len(pitch1), len(pitch2)
    cost = np.abs(pitch1[:, None] - p2_shifted[None, :])
    
    try:
        D, wp = librosa.sequence.dtw(C=cost)
        dfw_pitch2 = np.zeros(N, dtype=float)
        counts = np.zeros(N, dtype=float)
        for p1_idx, p2_idx in wp:
            dfw_pitch2[p1_idx] += p2_shifted[p2_idx]
            counts[p1_idx] += 1.0
        counts[counts == 0] = 1.0
        dfw_pitch2 /= counts
        dfw_cost = float(D[-1, -1] / len(wp))
        return dfw_pitch2, dfw_cost
    except Exception:
        return p2_shifted, 999.0


def compute_dtw_distance(pitch1: np.ndarray, pitch2: np.ndarray) -> Dict[str, float]:
    """
    Calculates Absolute DTW, Time-Frequency Aligned DTW, and 
    Matsumoto 1987 Dynamic Frequency Warping (DFW + DTW) dual alignment.
    """
    valid1 = pitch1[(pitch1 > 0) & (~np.isnan(pitch1))]
    valid2 = pitch2[(pitch2 > 0) & (~np.isnan(pitch2))]
    default_res = {
        "exact_norm": 999.0, "exact_raw": 999.0,
        "key_inv_exact_norm": 999.0, "key_inv_exact_raw": 999.0,
        "dfw_dtw_norm": 999.0, "dfw_dtw_raw": 999.0,
        "hz_exact_norm": 999.0, "hz_exact_raw": 999.0,
        "hz_key_inv_norm": 999.0, "hz_key_inv_raw": 999.0,
        "hz_dfw_dtw_norm": 999.0, "hz_dfw_dtw_raw": 999.0
    }
    if len(valid1) < 5 or len(valid2) < 5:
        return default_res
    
    # Downsample if sequences are long
    max_len = 1000
    if len(valid1) > max_len:
        valid1 = valid1[::int(len(valid1)/max_len)]
    if len(valid2) > max_len:
        valid2 = valid2[::int(len(valid2)/max_len)]
        
    res = dict(default_res)
    
    # 1. Exact Absolute DTW on MIDI semitones
    try:
        D, wp = librosa.sequence.dtw(valid1.reshape(1, -1), valid2.reshape(1, -1), metric='euclidean')
        res["exact_raw"] = float(D[-1, -1])
        res["exact_norm"] = float(D[-1, -1] / len(wp))
    except Exception:
        pass

    # 2. Key-Invariant DTW on MIDI semitones (Time & Frequency Aligned via Mean-Centering)
    try:
        v1_centered = valid1 - np.mean(valid1)
        v2_centered = valid2 - np.mean(valid2)
        D_ki, wp_ki = librosa.sequence.dtw(v1_centered.reshape(1, -1), v2_centered.reshape(1, -1), metric='euclidean')
        res["key_inv_exact_raw"] = float(D_ki[-1, -1])
        res["key_inv_exact_norm"] = float(D_ki[-1, -1] / len(wp_ki))
    except Exception:
        pass

    # 3. Matsumoto 1987 Dynamic Frequency Warping (DFW) + DTW Dual Alignment
    try:
        dfw_v2, dfw_cost = dynamic_frequency_warping(valid1, valid2)
        D_dfw, wp_dfw = librosa.sequence.dtw(valid1.reshape(1, -1), dfw_v2.reshape(1, -1), metric='euclidean')
        res["dfw_dtw_raw"] = float(D_dfw[-1, -1])
        res["dfw_dtw_norm"] = float(D_dfw[-1, -1] / len(wp_dfw))
    except Exception:
        pass

    # 4. Exact Absolute DTW on Hz Frequencies
    try:
        f0_1 = 440.0 * np.power(2.0, (valid1 - 69.0) / 12.0)
        f0_2 = 440.0 * np.power(2.0, (valid2 - 69.0) / 12.0)
        D_hz, wp_hz = librosa.sequence.dtw(f0_1.reshape(1, -1), f0_2.reshape(1, -1), metric='euclidean')
        res["hz_exact_raw"] = float(D_hz[-1, -1])
        res["hz_exact_norm"] = float(D_hz[-1, -1] / len(wp_hz))
        
        # Key-Invariant DTW on Hz Frequencies (Ratio / Mean-Centered in Hz)
        f0_1_norm = f0_1 / np.mean(f0_1)
        f0_2_norm = f0_2 / np.mean(f0_2)
        D_hz_ki, wp_hz_ki = librosa.sequence.dtw(f0_1_norm.reshape(1, -1), f0_2_norm.reshape(1, -1), metric='euclidean')
        res["hz_key_inv_raw"] = float(D_hz_ki[-1, -1])
        res["hz_key_inv_norm"] = float(D_hz_ki[-1, -1] / len(wp_hz_ki))
        
        # Matsumoto DFW on Hz Frequencies
        dfw_hz_2, _ = dynamic_frequency_warping(f0_1, f0_2)
        D_hz_dfw, wp_hz_dfw = librosa.sequence.dtw(f0_1.reshape(1, -1), dfw_hz_2.reshape(1, -1), metric='euclidean')
        res["hz_dfw_dtw_raw"] = float(D_hz_dfw[-1, -1])
        res["hz_dfw_dtw_raw"] = float(D_hz_dfw[-1, -1] / len(wp_hz_dfw))
    except Exception:
        pass

    return res


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
    """Safely converts numpy data types for JSON serialization."""
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ==============================================================================
# CACHE MANAGEMENT UTILITIES
# ==============================================================================

def get_phase_cache_dirs(base_cache_dir: Path) -> Dict[str, Path]:
    return {
        "phase1": base_cache_dir / "phase1_f0",
        "phase2": base_cache_dir / "phase2_matrices",
        "phase3": base_cache_dir / "phase3_results",
    }


def clear_cache_dir(target_dir: Path, description: str):
    """Safely removes a cache directory."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f" [CACHE FLUSH CIARP2] Eliminada carpeta de caché: {target_dir} ({description})")
    else:
        print(f" [CACHE INFO CIARP2] La carpeta {target_dir} ya estaba vacía.")


# ==============================================================================
# PHASE 1: FEATURE EXTRACTION (F0 PITCH & ENERGY)
# ==============================================================================

def run_phase1_extraction(
    all_files: Dict[str, Path],
    methods: List[str],
    base_cache_dir: Path,
    legacy_cache_dir: Path = Path("cache_ciarp")
) -> Dict[Tuple[str, str], MelodyFeatures]:
    """
    FASE 1 CIARP2: Extrae contornos de pitch (f0), tiempos y energía.
    Guarda los resultados en cache_ciarp2/phase1_f0/{method}/{filename}.json
    """
    print("\n" + "=" * 80)
    print(" FASE 1 CIARP2: Extracción y Caché de Características de Melodía (f0 Pitch)")
    print("=" * 80)
    
    p1_dir = get_phase_cache_dirs(base_cache_dir)["phase1"]
    p1_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_features = {}
    
    for method in methods:
        method_p1_dir = p1_dir / method
        method_p1_dir.mkdir(parents=True, exist_ok=True)
        
        legacy_method_dir = legacy_cache_dir / "phase1_f0" / method
        analyzer = MelodyAnalyzer(extraction_method=method)
        
        for file_key, file_path in all_files.items():
            cache_p1_file = method_p1_dir / f"{file_path.stem}.json"
            legacy_file = legacy_method_dir / f"{file_path.stem}.json"
            
            features = None
            
            # 1. Check Phase 1 cache
            if cache_p1_file.exists():
                try:
                    with open(cache_p1_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    features = MelodyFeatures.from_dict(data)
                except Exception:
                    pass
            
            # 2. Fallback to cache_ciarp directory
            if features is None and legacy_file.exists():
                try:
                    with open(legacy_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    features = MelodyFeatures.from_dict(data)
                except Exception:
                    pass
            
            # 3. Extract features directly if not cached
            if features is None:
                print(f"   [FASE 1 CIARP2 - Extrayendo] {method} | {file_path.name}...")
                target_sr = 16000 if method in ["rmvpe", "crepe", "bs_roformer_rmvpe", "demucs_rmvpe"] else analyzer.sample_rate
                audio, sr = librosa.load(str(file_path), sr=target_sr)
                features = extract_melody_features(audio, sr, method=method, hop_length=analyzer.hop_length, label=file_path.name)
            
            # Save into Phase 1 cache
            try:
                with open(cache_p1_file, 'w', encoding='utf-8') as f:
                    json.dump(features.to_dict(), f, default=safe_json)
            except Exception as e:
                print(f"Warning saving Phase 1 cache for {file_path.name}: {e}")
                
            extracted_features[(method, file_key, file_path.stem)] = features
            
        print(f" [FASE 1 COMPLETADA CIARP2] Método: {METHOD_DISPLAY_NAMES.get(method, method)}")
        
    return extracted_features


# ==============================================================================
# PHASE 2: REPRESENTATION & MATRIX GENERATION (SSM & TIME-FREQ ALIGNED DTW)
# ==============================================================================

def run_phase2_matrix_generation(
    common_keys: List[str],
    orig_files: Dict[str, Path],
    cover_files: Dict[str, Path],
    methods: List[str],
    base_cache_dir: Path,
    classifier: MelodyClassifierThesis
) -> Dict[str, Any]:
    """
    FASE 2 CIARP2: Lee f0 de Fase 1, realiza la segmentación formal (SSM / Algoritmo 1) y
    calcula la matriz de alineamiento DTW alineada en Tiempo y Frecuencia (Key-Invariant DTW).
    """
    print("\n" + "=" * 80)
    print(" FASE 2 CIARP2: Generación de SSM, Segmentación y DTW Alineado en Tiempo y Frecuencia")
    print("=" * 80)
    
    p1_dir = get_phase_cache_dirs(base_cache_dir)["phase1"]
    p2_dir = get_phase_cache_dirs(base_cache_dir)["phase2"]
    p2_dir.mkdir(parents=True, exist_ok=True)
    
    phase2_data = {}
    
    for method in methods:
        print(f"\n---> [FASE 2 CIARP2] Procesando Matrices y DTW Alineado para: {METHOD_DISPLAY_NAMES.get(method, method)}...")
        method_p2_dir = p2_dir / method
        method_p2_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer = MelodyAnalyzer(extraction_method=method, classifier=classifier)
        method_p1_dir = p1_dir / method
        
        pair_sequences = {}
        pair_pitches = {}
        pair_dtw_metrics = {}
        
        for key in common_keys:
            cache_pair_file = method_p2_dir / f"pair_{key}.json"
            
            # Check Phase 2 cache
            if cache_pair_file.exists():
                try:
                    with open(cache_pair_file, 'r', encoding='utf-8') as f:
                        p2_dict = json.load(f)
                    pair_sequences[key] = (p2_dict['seq_orig'], p2_dict['seq_cover'])
                    pair_pitches[key] = (np.array(p2_dict['pitch_orig']), np.array(p2_dict['pitch_cover']))
                    pair_dtw_metrics[key] = p2_dict['dtw_res']
                    continue
                except Exception:
                    pass
            
            # Load Phase 1 features for Original & Cover
            orig_path = orig_files[key]
            cover_path = cover_files[key]
            
            p1_orig_file = method_p1_dir / f"{orig_path.stem}.json"
            p1_cover_file = method_p1_dir / f"{cover_path.stem}.json"
            
            with open(p1_orig_file, 'r', encoding='utf-8') as f:
                feat_orig = MelodyFeatures.from_dict(json.load(f))
            with open(p1_cover_file, 'r', encoding='utf-8') as f:
                feat_cover = MelodyFeatures.from_dict(json.load(f))
                
            # Perform SSM Segmentation & Algorithm 1 State Classification
            res_orig = analyzer.analyze_features(feat_orig)
            res_cover = analyzer.analyze_features(feat_cover)
            
            seq_orig = [seg.label for seg in res_orig.segments]
            seq_cover = [seg.label for seg in res_cover.segments]
            
            # Compute Time & Frequency Aligned DTW
            dtw_res = compute_dtw_distance(res_orig.features.pitch_midi, res_cover.features.pitch_midi)
            
            pair_sequences[key] = (seq_orig, seq_cover)
            pair_pitches[key] = (res_orig.features.pitch_midi, res_cover.features.pitch_midi)
            pair_dtw_metrics[key] = dtw_res
            
            # Save into Phase 2 cache
            p2_dict = {
                "pair_key": key,
                "seq_orig": seq_orig,
                "seq_cover": seq_cover,
                "pitch_orig": res_orig.features.pitch_midi,
                "pitch_cover": res_cover.features.pitch_midi,
                "dtw_res": dtw_res
            }
            try:
                with open(cache_pair_file, 'w', encoding='utf-8') as f:
                    json.dump(p2_dict, f, default=safe_json)
            except Exception as e:
                print(f"Warning saving Phase 2 cache for pair {key}: {e}")
                
        phase2_data[method] = {
            "pair_sequences": pair_sequences,
            "pair_pitches": pair_pitches,
            "pair_dtw_metrics": pair_dtw_metrics,
            "common_keys": common_keys
        }
        print(f" [FASE 2 COMPLETADA CIARP2] Matrices y DTW alineado en Tiempo-Frecuencia listos para {METHOD_DISPLAY_NAMES.get(method, method)}")
        
    return phase2_data


# ==============================================================================
# PHASE 3: CLASSIFICATION & BENCHMARK REPORTING (CIARP2)
# ==============================================================================

def run_phase3_classification_reporting(
    phase2_data: Dict[str, Any],
    methods: List[str],
    output_dir: Path,
    base_cache_dir: Path
) -> Dict[str, Any]:
    """
    FASE 3 CIARP2: Genera tablas formateadas reportando tanto el DTW Absoluto
    como el DTW Alineado en Tiempo y Frecuencia (Key-Invariant DTW).
    """
    print("\n" + "=" * 80)
    print(" FASE 3 CIARP2: Resultados de Clasificador y Reporte DTW Alineado en Tiempo-Frecuencia")
    print("=" * 80)
    
    p3_dir = get_phase_cache_dirs(base_cache_dir)["phase3"]
    p3_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_results = {}
    
    for method in methods:
        print(f"\n---> [FASE 3 CIARP2] Calculando métricas para: {METHOD_DISPLAY_NAMES.get(method, method)}...")
        m_data = phase2_data[method]
        pair_sequences = m_data["pair_sequences"]
        pair_dtw_metrics = m_data["pair_dtw_metrics"]
        common_keys = m_data["common_keys"]
        
        nlcs_scores = []
        exact_dtw_norm_list = []
        key_inv_dtw_norm_list = []
        hz_exact_norm_list = []
        reciprocal_ranks = []
        top5_hits = []
        
        for key in common_keys:
            seq_orig, seq_cover = pair_sequences[key]
            nlcs = calculate_lcs_ciarp(seq_orig, seq_cover)
            nlcs_scores.append(nlcs * 100.0)
            
            dtw_res = pair_dtw_metrics[key]
            if dtw_res["exact_norm"] < 990:
                exact_dtw_norm_list.append(dtw_res["exact_norm"])
            if dtw_res["key_inv_exact_norm"] < 990:
                key_inv_dtw_norm_list.append(dtw_res["key_inv_exact_norm"])
            if dtw_res["hz_exact_norm"] < 990:
                hz_exact_norm_list.append(dtw_res["hz_exact_norm"])

        # Target cover retrieval ranking (Target Cover Last tie-breaking)
        for q_key in common_keys:
            _, q_seq_cover = pair_sequences[q_key]
            similarities = []
            for r_key in common_keys:
                r_seq_orig, _ = pair_sequences[r_key]
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
        
        avg_exact_norm = float(np.mean(exact_dtw_norm_list)) if exact_dtw_norm_list else 0.0
        avg_key_inv_norm = float(np.mean(key_inv_dtw_norm_list)) if key_inv_dtw_norm_list else 0.0
        avg_hz_exact_norm = float(np.mean(hz_exact_norm_list)) if hz_exact_norm_list else 0.0

        # Compute 95% Bootstrap Confidence Intervals (1000 resamples)
        nlcs_ci = compute_bootstrap_ci(nlcs_scores, n_bootstraps=1000)
        mrr_ci = compute_bootstrap_ci([r * 100.0 for r in reciprocal_ranks], n_bootstraps=1000)
        top5_ci = compute_bootstrap_ci([t * 100.0 for t in top5_hits], n_bootstraps=1000)
        key_inv_ci = compute_bootstrap_ci(key_inv_dtw_norm_list, n_bootstraps=1000)

        # Binary Classification Grid Search (Table 3)
        pairwise_lcs = []
        for q_key in common_keys:
            _, q_seq_cover = pair_sequences[q_key]
            for r_key in common_keys:
                r_seq_orig, _ = pair_sequences[r_key]
                sim = calculate_lcs_ciarp(r_seq_orig, q_seq_cover)
                is_correct = (q_key == r_key)
                pairwise_lcs.append((sim, is_correct))
                
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

        res_dict = {
            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "nlcs_percent": avg_nlcs,
            "nlcs_ci95": nlcs_ci,
            "mrr_percent": avg_mrr,
            "mrr_ci95": mrr_ci,
            "top5_percent": avg_top5,
            "top5_ci95": top5_ci,
            "raw_dtw_norm": avg_exact_norm,
            "key_inv_dtw_norm": avg_key_inv_norm,
            "key_inv_dtw_norm_ci95": key_inv_ci,
            "hz_dtw_norm": avg_hz_exact_norm,
            "num_pairs": len(nlcs_scores),
            "table3_metrics": best_metrics
        }
        benchmark_results[method] = res_dict
        
        # Save to Phase 3 cache
        p3_file = p3_dir / f"result_{method}.json"
        with open(p3_file, 'w', encoding='utf-8') as f:
            json.dump(res_dict, f, indent=2, default=safe_json)

    # Determine best values for bolding in Table 2
    best_nlcs_val = max(res['nlcs_percent'] for res in benchmark_results.values())
    best_dtw_val = min(res['key_inv_dtw_norm'] for res in benchmark_results.values())
    best_mrr_val = max(res['mrr_percent'] for res in benchmark_results.values())
    best_top5_val = max(res['top5_percent'] for res in benchmark_results.values())

    # Generate LaTeX Table 2
    latex_table2_path = output_dir / "ciarp2_table2_results.tex"
    with open(latex_table2_path, "w", encoding="utf-8") as f:
        f.write("% CIARP2 Table 2: CSI Performance Across Melodic Extraction Methods (Time & Frequency Aligned DTW)\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{CSI Performance Across Melodic Extraction Methods with Time-Frequency Aligned DTW.}\n")
        f.write("\\label{tab:ciarp2_table2}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Avg. NLCS (\\%)} & \\textbf{DTW (Time-Freq Aligned)} & \\textbf{MRR (\\%)} & \\textbf{Top-5 (\\%)} \\\\\n\\hline\n")
        for m, res in benchmark_results.items():
            disp = res['display_name']
            
            nlcs_val = res['nlcs_percent']
            nlcs_str = f"\\textbf{{{nlcs_val:.2f}}}" if abs(nlcs_val - best_nlcs_val) < 1e-4 else f"{nlcs_val:.2f}"
            
            dtw_val = res['key_inv_dtw_norm']
            dtw_str = f"\\textbf{{{dtw_val:.2f}}}" if abs(dtw_val - best_dtw_val) < 1e-4 else f"{dtw_val:.2f}"
            
            mrr_val = res['mrr_percent']
            mrr_val_str = f"\\textbf{{{mrr_val:.2f}}}" if abs(mrr_val - best_mrr_val) < 1e-4 else f"{mrr_val:.2f}"
            mrr_str = f"{mrr_val_str} [{res['mrr_ci95'][0]:.1f}, {res['mrr_ci95'][1]:.1f}]"
            
            top5_val = res['top5_percent']
            top5_val_str = f"\\textbf{{{top5_val:.2f}}}" if abs(top5_val - best_top5_val) < 1e-4 else f"{top5_val:.2f}"
            top5_str = f"{top5_val_str} [{res['top5_ci95'][0]:.1f}, {res['top5_ci95'][1]:.1f}]"
            
            f.write(f"{disp} & {nlcs_str} & {dtw_str} & {mrr_str} & {top5_str} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Generate LaTeX Table 3
    latex_table3_path = output_dir / "ciarp2_table3_results.tex"
    with open(latex_table3_path, "w", encoding="utf-8") as f:
        f.write("% CIARP2 Table 3: Binary classification threshold analysis and confusion matrix parameters\n")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Binary classification threshold analysis and confusion matrix parameters.}\n")
        f.write("\\label{tab:ciarp2_table3}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\n\\hline\n")
        f.write("\\textbf{Method} & \\textbf{Opt. Thresh} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{TP} & \\textbf{FP} & \\textbf{FN} & \\textbf{TN} \\\\\n\\hline\n")
        for m in methods:
            disp = benchmark_results[m]['display_name']
            bm = benchmark_results[m].get("table3_metrics", {})
            if bm:
                f.write(f"{disp} & {bm['thresh']:.4f} & {bm['f1']:.4f} & {bm['prec']:.4f} & {bm['rec']:.4f} & {bm['tp']} & {bm['fp']} & {bm['fn']} & {bm['tn']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Save JSON summary report
    json_report_path = output_dir / "ciarp2_benchmark_summary.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, default=safe_json)

    # Save ASCII summary table
    txt_table_path = output_dir / "ciarp2_summary_table.txt"
    lines = []
    lines.append("=" * 140)
    lines.append("CIARP2 BENCHMARK SUMMARY TABLE 2 (TIME & FREQUENCY ALIGNED DTW)")
    lines.append("=" * 140)
    lines.append(f"{'Method':<20} | {'Avg. NLCS (%) [95% CI]':<26} | {'DTW (Time-Freq Aligned) [95% CI]':<36} | {'MRR (%) [95% CI]':<22} | {'Top-5 (%) [95% CI]':<22}")
    lines.append("-" * 140)
    for m, res in benchmark_results.items():
        disp = res['display_name']
        nlcs_s = f"{res['nlcs_percent']:.2f} [{res['nlcs_ci95'][0]:.1f}, {res['nlcs_ci95'][1]:.1f}]"
        dtw_s = f"{res['key_inv_dtw_norm']:.2f} [{res['key_inv_dtw_norm_ci95'][0]:.1f}, {res['key_inv_dtw_norm_ci95'][1]:.1f}]"
        mrr_s = f"{res['mrr_percent']:.2f} [{res['mrr_ci95'][0]:.1f}, {res['mrr_ci95'][1]:.1f}]"
        top5_s = f"{res['top5_percent']:.2f} [{res['top5_ci95'][0]:.1f}, {res['top5_ci95'][1]:.1f}]"
        lines.append(f"{disp:<20} | {nlcs_s:<26} | {dtw_s:<36} | {mrr_s:<22} | {top5_s:<22}")
    lines.append("=" * 140)
    
    lines.append("\n" + "=" * 95)
    lines.append("CIARP2 BENCHMARK SUMMARY TABLE 3 (BINARY CLASSIFICATION)")
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

    with open(latex_table2_path, "r", encoding="utf-8") as f:
        latex_t2_str = f.read()
    with open(latex_table3_path, "r", encoding="utf-8") as f:
        latex_t3_str = f.read()

    print(f"\n========================================================")
    print(f" BENCHMARK CIARP2 COMPLETO - TABLAS DE RESULTADOS")
    print(f"========================================================\n")
    print("--- LATEX TABLE 2 (TIME-FREQUENCY ALIGNED DTW) ---")
    print(latex_t2_str)
    print("--- LATEX TABLE 3 (BINARY CLASSIFICATION ANALYSIS) ---")
    print(latex_t3_str)
    print("--- TEXT SUMMARY TABLE ---")
    print(txt_table_content)
    print(f" Archivo LaTeX Tabla 2 guardado en: {latex_table2_path}")
    print(f" Archivo LaTeX Tabla 3 guardado en: {latex_table3_path}")
    print(f" Archivo Texto Resumen guardado en: {txt_table_path}")
    print(f" Archivo JSON Resumen guardado en: {json_report_path}")
    print(f"========================================================\n")
    
    return benchmark_results


# ==============================================================================
# MAIN EXECUTOR CIARP2
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="MC-MSA CIARP2 Benchmark Runner (Time & Frequency Aligned DTW)")
    parser.add_argument("--orig-dir", type=str, default="dataset_OA/originales", help="Path to originals folder")
    parser.add_argument("--cover-dir", type=str, default="dataset_OA/covers", help="Path to covers folder")
    parser.add_argument("--output-dir", type=str, default="outputs_ciarp2", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default="cache_ciarp2", help="Path to CIARP2 cache directory")
    parser.add_argument("--methods", nargs="+", default=CIARP_EVAL_METHODS, help="Extraction methods to evaluate")
    parser.add_argument("--phase", type=str, choices=["1", "2", "3", "all"], default="all", help="Phase to execute (1, 2, 3, or all)")
    
    parser.add_argument("--clear-phase1", action="store_true", help="Flush Phase 1 cache (f0 features)")
    parser.add_argument("--clear-phase2", action="store_true", help="Flush Phase 2 cache (SSM & DTW matrices)")
    parser.add_argument("--clear-phase3", action="store_true", help="Flush Phase 3 cache (Classification results)")
    parser.add_argument("--clear-all-cache", action="store_true", help="Flush ALL caches")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive menu")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    orig_path = (script_dir / args.orig_dir).resolve()
    cover_path = (script_dir / args.cover_dir).resolve()
    output_path = (script_dir / args.output_dir).resolve()
    base_cache_path = (script_dir / args.cache_dir).resolve()
    
    phase_dirs = get_phase_cache_dirs(base_cache_path)
    
    if args.interactive:
        print("\n========================================================")
        print(" MC-MSA CIARP2 (Time & Frequency Aligned DTW) - Caché")
        print("========================================================")
        print(" [1] Borrar Caché FASE 1 (Extracción de F0)")
        print(" [2] Borrar Caché FASE 2 (SSM y DTW Alineado)")
        print(" [3] Borrar Caché FASE 3 (Resultados y Tablas CIARP2)")
        print(" [4] Borrar TODO el Caché CIARP2")
        print(" [5] Ejecutar Fase 1 (Extracción de F0)")
        print(" [6] Ejecutar Fase 2 (SSM y DTW Alineado Tiempo-Frecuencia)")
        print(" [7] Ejecutar Fase 3 (Clasificador y Tablas CIARP2)")
        print(" [8] Ejecutar Pipeline COMPLETO (Fases 1 -> 2 -> 3)")
        print(" [0] Salir")
        print("========================================================")
        choice = input("Selecciona una opción [0-8]: ").strip()
        
        if choice == "1":
            clear_cache_dir(phase_dirs["phase1"], "Fase 1: f0 pitch")
            return
        elif choice == "2":
            clear_cache_dir(phase_dirs["phase2"], "Fase 2: SSM & DTW matrices")
            return
        elif choice == "3":
            clear_cache_dir(phase_dirs["phase3"], "Fase 3: Clasificador")
            return
        elif choice == "4":
            clear_cache_dir(base_cache_path, "TODAS las fases")
            return
        elif choice == "5":
            args.phase = "1"
        elif choice == "6":
            args.phase = "2"
        elif choice == "7":
            args.phase = "3"
        elif choice == "8":
            args.phase = "all"
        else:
            print("Saliendo...")
            return

    # Handle CLI cache clearing flags
    if args.clear_all_cache:
        clear_cache_dir(base_cache_path, "TODAS las fases")
    if args.clear_phase1:
        clear_cache_dir(phase_dirs["phase1"], "Fase 1: f0 pitch")
    if args.clear_phase2:
        clear_cache_dir(phase_dirs["phase2"], "Fase 2: SSM & DTW matrices")
    if args.clear_phase3:
        clear_cache_dir(phase_dirs["phase3"], "Fase 3: Clasificador")

    orig_files = get_audio_files(orig_path)
    cover_files = get_audio_files(cover_path)
    common_keys = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
    
    all_audio_files = {}
    for k in common_keys:
        all_audio_files[f"orig_{k}"] = orig_files[k]
        all_audio_files[f"cover_{k}"] = cover_files[k]

    classifier = MelodyClassifierThesis(slope_epsilon=CIARP_THETA_SLOPE, energy_tau=CIARP_THETA_ENERGY)

    # EXECUTE PHASE 1
    if args.phase in ["1", "all"]:
        run_phase1_extraction(all_audio_files, args.methods, base_cache_path)
        if args.phase == "1":
            print("\n[FASE 1 COMPLETADA CIARP2] Las características f0 se han guardado en caché.")
            return

    # EXECUTE PHASE 2
    phase2_data = None
    if args.phase in ["2", "3", "all"]:
        phase2_data = run_phase2_matrix_generation(common_keys, orig_files, cover_files, args.methods, base_cache_path, classifier)
        if args.phase == "2":
            print("\n[FASE 2 COMPLETADA CIARP2] Las matrices SSM y DTW alineado en Tiempo-Frecuencia se han guardado en caché.")
            return

    # EXECUTE PHASE 3
    if args.phase in ["3", "all"]:
        if phase2_data is None:
            print("Error: Se requiere ejecutar o cargar Fase 2 para ejecutar Fase 3.")
            return
        run_phase3_classification_reporting(phase2_data, args.methods, output_path, base_cache_path)


if __name__ == "__main__":
    main()
