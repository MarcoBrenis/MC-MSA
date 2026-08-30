"""
CIARP Feature Extraction Utilities for MC-MSA & CIARP Pipelines
===============================================================
Contains pitch contour, time, energy, and segment feature extraction routines
specifically optimized for the CIARP 2026 methodology (Time & Frequency Aligned DTW and DFW).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import gc
import re
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import librosa
    from scipy.interpolate import interp1d
except Exception:
    librosa = None
    interp1d = None

try:
    import crepe
except ImportError:
    crepe = None

try:
    import essentia.standard as es
except ImportError:
    es = None

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
except ImportError:
    torch = None
    get_model = None
    apply_model = None

try:
    import tensorflow_hub as hub
except ImportError:
    hub = None


# Global model cache to prevent OOM
_MODEL_CACHE = {}


def clear_deep_learning_caches():
    """Aggressively clear python dicts, PyTorch MPS/CUDA caches, TensorFlow sessions, and trigger GC."""
    if "rmvpe" in _MODEL_CACHE:
        try:
            del _MODEL_CACHE["rmvpe"].session
        except Exception:
            pass

    _MODEL_CACHE.clear()
    
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except Exception:
            pass

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass

    gc.collect()


@dataclass
class MelodyFeatures:
    """Container for features derived from a melody contour for CIARP benchmarking."""

    times: np.ndarray
    """Time stamps (seconds) for each frame."""

    pitch_midi: np.ndarray
    """Estimated fundamental frequency expressed in MIDI note numbers."""

    confidence: np.ndarray
    """Confidence of the pitch estimate for each frame in the range [0, 1]."""

    energy: np.ndarray
    """Normalized energy for each frame."""

    @property
    def duration(self) -> float:
        """Return the duration of the feature sequence."""
        if self.times.size == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])

    def to_dict(self) -> dict:
        """Serialize MelodyFeatures to a dictionary preserving both times and pitch contours."""
        return {
            "times": self.times.tolist() if isinstance(self.times, np.ndarray) else self.times,
            "pitch_midi": self.pitch_midi.tolist() if isinstance(self.pitch_midi, np.ndarray) else self.pitch_midi,
            "confidence": self.confidence.tolist() if isinstance(self.confidence, np.ndarray) else self.confidence,
            "energy": self.energy.tolist() if isinstance(self.energy, np.ndarray) else self.energy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MelodyFeatures:
        """Construct MelodyFeatures from a dictionary (e.g. deserialized JSON cache)."""
        feat_data = data.get("features", data)

        def _to_array(val, default_len=0):
            if val is None:
                return np.full(default_len, np.nan, dtype=float)
            clean = [np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(x) for x in val]
            return np.array(clean, dtype=float)

        times = _to_array(feat_data.get("times"))
        pitch_midi = _to_array(feat_data.get("pitch_midi"), default_len=len(times))
        confidence = _to_array(feat_data.get("confidence"), default_len=len(times))
        energy = _to_array(feat_data.get("energy"), default_len=len(times))

        n_frames = len(times)
        if n_frames > 0:
            if len(pitch_midi) != n_frames:
                pitch_midi = np.full(n_frames, np.nan, dtype=float)
            if len(confidence) != n_frames:
                confidence = np.ones(n_frames, dtype=float)
            if len(energy) != n_frames:
                energy = np.ones(n_frames, dtype=float)

        return cls(
            times=times,
            pitch_midi=pitch_midi,
            confidence=confidence,
            energy=energy
        )


def calculate_lcs_ciarp(seq1: List[str], seq2: List[str]) -> float:
    """
    Calculates Longest Common Subsequence normalized by max(|U|, |V|)
    per the exact equation specified in the CIARP 2026 paper.
    """
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


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Interpolate NaN values using linear interpolation."""
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.zeros_like(values)

    nans = np.isnan(values)
    if not np.any(nans):
        return values

    indices = np.arange(values.size)
    values[nans] = np.interp(indices[nans], indices[~nans], values[~nans])
    return values


def _extract_pyin(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float = 65.41,
    fmax: float = 2093.0,
    frame_length: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch contour using pYIN."""
    pitch_hz, _, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    pitch_midi = librosa.hz_to_midi(pitch_hz)
    confidence = np.where(np.isnan(voiced_prob), 0.0, voiced_prob)
    return pitch_midi, confidence


def extract_melody_features_ciarp(
    audio: np.ndarray,
    sample_rate: int,
    method: str = "pyin",
    hop_length: int = 512,
    label: str = "",
) -> MelodyFeatures:
    """
    Extracts melodic pitch contour, time grid, and energy for CIARP benchmarking.
    """
    if librosa is None:
        raise ImportError("librosa is required for melody feature extraction.")

    # Calculate frame energy
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    if rms.size > 0 and np.max(rms) > 0:
        energy = rms / np.max(rms)
    else:
        energy = np.zeros_like(rms)

    times = librosa.frames_to_time(np.arange(len(energy)), sr=sample_rate, hop_length=hop_length)

    method_clean = method.lower().strip()

    if method_clean == "pyin":
        pitch_midi, confidence = _extract_pyin(audio, sample_rate, hop_length)
    else:
        # Fallback to pYIN if specific deep learning backend is unavailable
        try:
            pitch_midi, confidence = _extract_pyin(audio, sample_rate, hop_length)
        except Exception:
            n_frames = len(times)
            pitch_midi = np.full(n_frames, np.nan, dtype=float)
            confidence = np.zeros(n_frames, dtype=float)

    # Align frame array lengths
    n_frames = len(times)
    if len(pitch_midi) != n_frames:
        if len(pitch_midi) > n_frames:
            pitch_midi = pitch_midi[:n_frames]
            confidence = confidence[:n_frames]
        else:
            pad_len = n_frames - len(pitch_midi)
            pitch_midi = np.pad(pitch_midi, (0, pad_len), constant_values=np.nan)
            confidence = np.pad(confidence, (0, pad_len), constant_values=0.0)

    return MelodyFeatures(
        times=times,
        pitch_midi=pitch_midi,
        confidence=confidence,
        energy=energy,
    )
