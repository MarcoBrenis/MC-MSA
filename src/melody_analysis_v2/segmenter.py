"""Segmentation logic for melody analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

from .config import (
    CHECKERBOARD_RADIUS,
    KERNEL_SIZE,
    PEAK_THRESHOLD,
    MIN_SEPARATION,
    NUMERICAL_EPSILON,
)
from .features import MelodyFeatures


@dataclass
class MelodySegment:
    """Representation of a temporal segment within the melody."""

    start_time: float
    end_time: float
    start_index: int
    end_index: int

    def duration(self) -> float:
        """Duration of the segment in seconds."""

        return float(self.end_time - self.start_time)


class MelodySegmenter:
    """Detects structural boundaries within a melody contour.

    The segmentation strategy is inspired by novelty detection techniques employed
    in MSAF but adjusted to work with melodic descriptors. The algorithm computes
    a checkerboard-convolved self-similarity matrix (pitch + energy) and selects
    salient peaks of the resulting novelty curve as boundaries.
    """

    def __init__(
        self,
        *,
        kernel_size: int = KERNEL_SIZE,
        peak_threshold: float = PEAK_THRESHOLD,
        min_separation: int = MIN_SEPARATION,
        use_self_similarity: bool = True,
        checkerboard_radius: int = CHECKERBOARD_RADIUS,
        max_ssm_frames: int = 3000,
        filter_type: str = "gaussian",
    ) -> None:
        """Create a segmenter.

        Parameters
        ----------
        kernel_size:
            Standard deviation/window size of kernel applied to novelty curve.
        peak_threshold:
            Minimum relative height (0-1) for peaks to be considered boundaries.
        filter_type:
            Type of smoothing filter: 'gaussian', 'median', or 'hybrid'.
        """

        self.kernel_size = kernel_size
        self.peak_threshold = peak_threshold
        self.min_separation = min_separation
        self.use_self_similarity = use_self_similarity
        self.checkerboard_radius = checkerboard_radius
        self.max_ssm_frames = max_ssm_frames
        self.filter_type = filter_type
        self.last_step = 1

    def compute_self_similarity(self, features: MelodyFeatures) -> np.ndarray:
        """Compute a cosine self-similarity matrix from pitch and energy."""
        if getattr(features, '_sim_cache', None) is not None:
            return features._sim_cache

        stacked = np.vstack((features.pitch_midi, features.energy)).T.astype(np.float32)
        mean = np.mean(stacked, axis=0, keepdims=True)
        std = np.std(stacked, axis=0, keepdims=True) + 1e-40
        stacked = (stacked - mean) / std
        
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        normalized = stacked / np.maximum(norms, 1e-40)

        sim = (normalized @ normalized.T).astype(np.float32)
        sim = (sim + 1.0) / 2.0
        np.fill_diagonal(sim, 1.0)
        features._sim_cache = sim
        return sim

    def compute_checkerboard_novelty(self, sim: np.ndarray) -> np.ndarray:
        """Compute novelty along the diagonal of the self-similarity matrix using 2D integral image."""

        r = self.checkerboard_radius
        n = sim.shape[0]
        if n == 0 or n < 2 * r:
            return np.zeros(n, dtype=float)

        # 2D Integral Image (Summed-Area Table) with 0-padding at top and left
        sat = np.pad(np.cumsum(np.cumsum(sim, axis=0), axis=1), ((1, 0), (1, 0)))

        i = np.arange(r, n - r)

        # Quadrant sums for all frames i simultaneously
        tl = sat[i, i] - sat[i - r, i] - sat[i, i - r] + sat[i - r, i - r]
        tr = sat[i, i + r] - sat[i - r, i + r] - sat[i, i] + sat[i - r, i]
        bl = sat[i + r, i] - sat[i, i] - sat[i + r, i - r] + sat[i, i - r]
        br = sat[i + r, i + r] - sat[i, i + r] - sat[i + r, i] + sat[i, i]

        novelty = np.zeros(n, dtype=float)
        novelty[r : n - r] = (tl - tr - bl + br).astype(float)

        novelty = np.maximum(novelty, 0.0)
        max_val = np.max(novelty)
        if max_val > 0:
            novelty = novelty / max_val

        # Apply smooth filtering: gaussian, median, or hybrid (median + gaussian)
        filter_type = getattr(self, 'filter_type', 'gaussian')
        kernel_sz = int(self.kernel_size)
        if kernel_sz % 2 == 0:
            kernel_sz += 1  # Median filter requires odd kernel size

        if filter_type == 'median':
            novelty = median_filter(novelty, size=kernel_sz)
        elif filter_type == 'hybrid':
            novelty = median_filter(novelty, size=kernel_sz)
            novelty = gaussian_filter1d(novelty, sigma=self.kernel_size)
        else:  # default gaussian
            novelty = gaussian_filter1d(novelty, sigma=self.kernel_size)

        return novelty

    def compute_novelty(
        self, features: MelodyFeatures, *, return_components: bool = False
    ):
        """Compute the novelty curve used for segmentation.

        Parameters
        ----------
        features:
            Extracted melodic descriptors.
        return_components:
            When ``True`` returns the combined novelty along with the base
            derivative novelty, the SSM-derived novelty (or ``None``), and the
            self-similarity matrix used to compute it (or ``None``).
        """

        pitch = features.pitch_midi
        energy = features.energy

        pitch_diff = np.abs(np.diff(pitch, prepend=pitch[0]))
        energy_diff = np.abs(np.diff(energy, prepend=energy[0]))

        if np.max(pitch_diff) > 0:
            pitch_diff = pitch_diff / np.max(pitch_diff)
        if np.max(energy_diff) > 0:
            energy_diff = energy_diff / np.max(energy_diff)

        base_novelty = pitch_diff + energy_diff
        base_novelty = gaussian_filter1d(base_novelty, sigma=self.kernel_size)

        if not self.use_self_similarity:
            if return_components:
                return base_novelty, base_novelty, None, None
            return base_novelty

        sim = self.compute_self_similarity(features)
        ssm_novelty = self.compute_checkerboard_novelty(sim)

        # In the thesis version, boundary detection relies solely on the global SSM novelty
        # to simplify explanations and match the thesis manuscript.
        combined = ssm_novelty
        if np.max(combined) > 0:
            combined = combined / np.max(combined)

        if return_components:
            return combined, base_novelty, ssm_novelty, sim
        return combined

    def find_boundaries(self, novelty: np.ndarray) -> np.ndarray:
        """Locate peaks in the novelty curve."""

        if novelty.size == 0:
            return np.array([], dtype=int)

        if np.max(novelty) > 0:
            height = self.peak_threshold * np.max(novelty)
        else:
            height = self.peak_threshold

        peaks, _ = find_peaks(novelty, height=height, distance=self.min_separation)
        return peaks.astype(int)

    def segment(self, features: MelodyFeatures) -> List[MelodySegment]:
        """Segment the melody based on extracted features with adaptive downsampling."""
        
        n_frames = len(features.times)
        self.last_step = 1
        
        if n_frames > self.max_ssm_frames:
            self.last_step = int(np.ceil(n_frames / self.max_ssm_frames))
            # Downsample features for structural analysis (SSM/Novelty)
            ds_features = MelodyFeatures(
                times=features.times[::self.last_step],
                pitch_midi=features.pitch_midi[::self.last_step],
                confidence=features.confidence[::self.last_step],
                energy=features.energy[::self.last_step]
            )
        else:
            ds_features = features

        novelty, base_novelty, ssm_novelty, sim = self.compute_novelty(
            ds_features, return_components=True
        )
        
        # Store metadata for visualization and classification
        self.last_novelty = novelty
        self.last_base_novelty = base_novelty
        self.last_ssm_novelty = ssm_novelty
        self.last_self_similarity = sim
        
        boundaries = self.find_boundaries(novelty)

        # Map downsampled boundaries back to original high-res indices
        frame_indices = [0] + (boundaries * self.last_step).tolist() + [len(features.times) - 1]
        segments: List[MelodySegment] = []
        for start, end in zip(frame_indices[:-1], frame_indices[1:]):
            start_idx = int(start)
            end_idx = int(end)
            if end_idx <= start_idx:
                continue
            segment = MelodySegment(
                start_time=float(features.times[start_idx]),
                end_time=float(features.times[end_idx]),
                start_index=start_idx,
                end_index=end_idx,
            )
            if segment.duration() <= 0:
                continue
            segments.append(segment)

        return segments


__all__ = ["MelodySegment", "MelodySegmenter"]
