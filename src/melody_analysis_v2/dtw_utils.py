"""
DTW & Dynamic Frequency Warping (DFW) Utilities
================================================
Implements Dynamic Time Warping (DTW), Key-Invariant DTW, and 
Matsumoto 1987 Dynamic Frequency Warping (DFW) dual alignment for melody contours.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def dynamic_frequency_warping(pitch1: np.ndarray, pitch2: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Implements Dynamic Frequency Warping (DFW, Matsumoto 1987 JASA).
    Aligns pitch contours non-linearly along the frequency/pitch axis (dual of DTW).
    """
    if librosa is None or len(pitch1) == 0 or len(pitch2) == 0:
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
    if librosa is None or len(valid1) < 5 or len(valid2) < 5:
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
        res["hz_dfw_dtw_norm"] = float(D_hz_dfw[-1, -1] / len(wp_hz_dfw))
    except Exception:
        pass

    return res
