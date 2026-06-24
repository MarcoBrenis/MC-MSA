"""Strict 3-class classifier (A, C, X) for CLEI paper experiments."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment
from .classifier import MelodySegmentAnnotation

class MelodyClassifierPaper:
    """
    Implements a strict 3-class (A, C, X) classification logic based on Caplin's theory.
    Optimized for RMVPE data (f0, energy, and voicing probability).
    """

    def __init__(
        self,
        *,
        min_voicing_thresh: float = 0.0,
        slope_epsilon: float = 0.15,
        energy_tau: float = 0.3,
    ) -> None:
        self.min_voicing_thresh = min_voicing_thresh
        self.slope_epsilon = slope_epsilon
        self.energy_tau = energy_tau

    def _safe_polyfit(self, x: np.ndarray, y: np.ndarray, deg: int = 1) -> float:
        """Calculates linear slope safely handling empty arrays and NaN."""
        if x.size < 2 or y.size < 2:
            return 0.0
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return 0.0
        try:
            coeffs = np.polyfit(x[mask], y[mask], deg)
            return float(coeffs[0])
        except Exception:
            return 0.0

    def classify(
        self, features: MelodyFeatures, segments: List[MelodySegment], sim_matrix: Optional[np.ndarray] = None, ssm_step: int = 1
    ) -> List[MelodySegmentAnnotation]:
        
        annotations: List[MelodySegmentAnnotation] = []
        if not segments:
            return annotations

        for segment in segments:
            idx = slice(segment.start_index, segment.end_index + 1)
            pitch = features.pitch_midi[idx]
            energy = features.energy[idx]
            times = features.times[idx]
            voicing = features.confidence[idx]
            
            # 1. Class 'X' (Noise/Silence)
            mean_voicing = float(np.mean(voicing)) if voicing.size > 0 else 0.0
            mean_energy = float(np.mean(energy)) if energy.size > 0 else 0.0
            
            # Identify voiced frames (confidence > 0.1, valid MIDI pitch > 0)
            voiced_mask = (voicing > 0.1) & (~np.isnan(pitch)) & (pitch > 0)
            proportion_voiced = float(np.mean(voiced_mask)) if voicing.size > 0 else 0.0
            
            # Calculate tail characteristics first so we can use them in the silence check
            tail_len = max(1, int(len(pitch) * 0.2))
            tail_idx = slice(-tail_len, None)
            
            tail_pitch = pitch[tail_idx]
            tail_times = times[tail_idx]
            tail_energy = float(np.mean(energy[tail_idx])) if energy[tail_idx].size > 0 else 0.0
            
            voiced_tail_mask = ~np.isnan(tail_pitch) & (tail_pitch > 0)
            if np.sum(voiced_tail_mask) >= 2:
                slope = self._safe_polyfit(tail_times[voiced_tail_mask], tail_pitch[voiced_tail_mask], 1)
            else:
                slope = 0.0

            # Simplified 3-rule Algorithm logic
            theta_silence = 0.08
            theta_slope = 0.0
            theta_energy = 0.15
            
            if mean_energy < theta_silence:
                label = "Silence"
                confidence = 1.0 - mean_voicing
                descriptor = {
                    "mean_voicing": mean_voicing,
                    "mean_energy": mean_energy,
                    "proportion_voiced": proportion_voiced,
                    "reason": f"Mean energy below threshold ({mean_energy:.3f} < {theta_silence:.2f})"
                }
            elif slope <= theta_slope and tail_energy < theta_energy:
                label = "Consequent"
                confidence = 0.8
                descriptor = {"f0_slope": slope, "energy_tail": tail_energy}
            else:
                label = "Antecedent"
                confidence = 0.8
                descriptor = {"f0_slope": slope, "energy_tail": tail_energy}

            annotations.append(
                MelodySegmentAnnotation(
                    segment=segment,
                    label=label,
                    confidence=confidence,
                    descriptor=descriptor,
                )
            )

        return annotations

def calculate_lcs(seq1: List[str], seq2: List[str]) -> float:
    """Calculates the Longest Common Subsequence similarity between two sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_len = dp[n][m]
    # Similarity score: 2 * LCS / (len1 + len2) or just normalized by max?
    # Common implementation is 2 * LCS / (len1 + len2)
    return 2.0 * lcs_len / (n + m)
