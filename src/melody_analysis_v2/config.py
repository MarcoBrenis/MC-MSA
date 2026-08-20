"""
Centralized Configuration and Single Source of Truth for MC-MSA Hyperparameters.
All modules and experiment runners import default hyperparameter values from here.
"""

from dataclasses import dataclass
from typing import Tuple

# Audio Signal Extraction Defaults
FS: int = 44100               # Audio sampling rate in Hz
HOP_LENGTH: int = 441         # Hop length (10 ms at 44.1 kHz)
RMS_FRAME_LENGTH: int = 2048  # Frame length for RMS energy computation

# Structural Segmentation Defaults
CHECKERBOARD_RADIUS: int = 8  # L: Radius of 2D Gaussian checkerboard kernel
KERNEL_SIZE: int = 2          # σ: Standard deviation for 1D Gaussian smoothing
PEAK_THRESHOLD: float = 0.20  # τ_peak: Minimum peak height threshold
MIN_SEPARATION: int = 10      # d_min: Minimum frame separation between boundaries
W_BASE: Tuple[float, float] = (0.9, 0.1)      # W_base: Weights for pitch and energy derivatives
W_COMBINED: Tuple[float, float] = (0.6, 0.4)  # W_combined: Weights for SSM novelty and base novelty fusion

# Functional Classification Defaults
MIN_VOICING_THRESH: float = 0.0 # v_bar: Minimum voicing threshold for silence
TAIL_PROPORTION: float = 0.20   # L_tail (p): Fraction of segment tail analyzed (20%)
SLOPE_EPSILON: float = 0.15     # ε_slope: F0 slope threshold for contour classification
ENERGY_TAU: float = 0.30        # τ_E: Energy threshold for cadence resolution

# Numerical Stability
NUMERICAL_EPSILON: float = 1e-40  # Epsilon for Z-score and L2 normalization


@dataclass
class MCMSAHyperparameters:
    """Dataclass holding all MC-MSA hyperparameters in one place."""
    sample_rate: int = FS
    hop_length: int = HOP_LENGTH
    rms_frame_length: int = RMS_FRAME_LENGTH
    checkerboard_radius: int = CHECKERBOARD_RADIUS
    kernel_size: int = KERNEL_SIZE
    peak_threshold: float = PEAK_THRESHOLD
    min_separation: int = MIN_SEPARATION
    w_base: Tuple[float, float] = W_BASE
    w_combined: Tuple[float, float] = W_COMBINED
    min_voicing_thresh: float = MIN_VOICING_THRESH
    tail_proportion: float = TAIL_PROPORTION
    slope_epsilon: float = SLOPE_EPSILON
    energy_tau: float = ENERGY_TAU
    numerical_epsilon: float = NUMERICAL_EPSILON
