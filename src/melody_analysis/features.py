"""Feature extraction utilities for melody analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency loaded lazily in tests
    import librosa
except Exception:  # pragma: no cover - handled gracefully
    librosa = None  # type: ignore


@dataclass
class MelodyFeatures:
    """Container for features derived from a melody contour."""

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


def extract_melody_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 1000.0,
    frame_length: int = 2048,
) -> MelodyFeatures:
    """Estimate melody features from a raw audio signal.

    Parameters
    ----------
    audio:
        Audio samples.
    sample_rate:
        Sampling rate of ``audio``.
    hop_length:
        Hop length used for feature extraction.
    fmin, fmax:
        Frequency range used for pitch estimation.
    frame_length:
        Frame length employed by the RMS computation.

    Returns
    -------
    MelodyFeatures
        Extracted time, pitch, confidence, and energy trajectories.
    """

    if librosa is None:
        raise ImportError(
            "librosa is required for feature extraction but is not available."
        )

    audio = np.asarray(audio, dtype=float)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Normalize audio to avoid numerical issues with pitch extraction.
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Estimate fundamental frequency using probabilistic YIN.
    pitch_hz, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # Convert to MIDI representation and interpolate unvoiced frames.
    pitch_midi = librosa.hz_to_midi(pitch_hz)
    pitch_midi = _interpolate_nans(pitch_midi)

    confidence = np.where(np.isnan(voiced_prob), 0.0, voiced_prob)

    # Compute energy using RMS and align to pitch frames.
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    ).flatten()

    energy = _interpolate_nans(energy)
    if energy.max() > 0:
        energy = energy / energy.max()

    times = librosa.frames_to_time(
        np.arange(len(pitch_midi)),
        sr=sample_rate,
        hop_length=hop_length,
    )

    return MelodyFeatures(times=times, pitch_midi=pitch_midi, confidence=confidence, energy=energy)


__all__ = ["MelodyFeatures", "extract_melody_features"]
