"""Strict 3-class classifier (A, C, X) for the original thesis/paper experiments."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment
from .classifier import MelodySegmentAnnotation

class MelodyClassifierThesis:
    """
    Implements a strict 3-class (A, C, X) classification logic based on Caplin's theory.
    Optimized for RMVPE data (f0, energy, and voicing probability).
    """

    def __init__(
        self,
        *,
        slope_epsilon: float = 0.15,
        energy_tau: float = 0.3,
        tail_proportion: float = 0.2,
    ) -> None:
        self.slope_epsilon = slope_epsilon
        self.energy_tau = energy_tau
        self.tail_proportion = tail_proportion

    def _safe_polyfit(self, x: np.ndarray, y: np.ndarray, deg: int = 1) -> float:
        """Calculates linear slope safely handling empty arrays and NaN."""
        if x.size < 2 or y.size < 2:
            return 0.0
        mask = (y > 0) & (~np.isnan(y))
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
            
            # 1. Class 'Silence' (if no valid f0 pitch in segment)
            valid_pitch_mask = (pitch > 0) & (~np.isnan(pitch))
            
            if np.sum(valid_pitch_mask) < 2:
                label = "Silence"
                confidence = 1.0
                descriptor = {"f0_slope": 0.0, "energy_tail": 0.0}
            else:
                # Analyze the tail of the segment
                tail_len = max(1, int(len(pitch) * self.tail_proportion))
                tail_idx = slice(-tail_len, None)
                
                tail_pitch = pitch[tail_idx]
                tail_times = times[tail_idx]
                tail_energy = float(np.mean(energy[tail_idx])) if energy[tail_idx].size > 0 else 0.0
                
                # Filter voiced frames in the tail for slope computation
                voiced_tail_mask = ~np.isnan(tail_pitch)
                if np.sum(voiced_tail_mask) >= 2:
                    slope = self._safe_polyfit(tail_times[voiced_tail_mask], tail_pitch[voiced_tail_mask], 1)
                else:
                    slope = 0.0
                
                # Strict Caplin formal function classification (A, C, X)
                # Antecedent (A): Ascending pitch slope OR high tail energy (open tension)
                # Consequent (C): Descending pitch slope AND low tail energy (harmonic resolution)
                # Continuation/Ambiguous (X): Neutral slope and moderate energy
                
                if slope > self.slope_epsilon:
                    label = "Antecedent"
                elif slope < -self.slope_epsilon and tail_energy < self.energy_tau:
                    label = "Consequent"
                elif tail_energy > (self.energy_tau * 1.5):
                    label = "Antecedent"
                elif tail_energy < (self.energy_tau * 0.5):
                    label = "Consequent"
                else:
                    label = "Continuation"  # Class 'X' (Neutral/Ambiguous)
                
                descriptor = {"f0_slope": slope, "energy_tail": tail_energy}
                confidence = 0.8  # Fixed confidence for logic-based labels

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
    
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    
    for x in seq1:
        for j in range(1, m + 1):
            if x == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)
    
    return 2.0 * prev[m] / (n + m)


# Alias definitions for CIARP 2026 / CLEI paper classifier compatibility
MelodyClassifierCIARP = MelodyClassifierThesis
MelodyClassifierPaper = MelodyClassifierThesis

