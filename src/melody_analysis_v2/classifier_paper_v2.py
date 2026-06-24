"""Strict 3-class classifier v2.0 (A, C, X) implementing corrected thesis rules."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment
from .classifier import MelodySegmentAnnotation

class MelodyClassifierPaperV2:
    """
    Implements the corrected strict 3-class (A, C, X) classification logic for MC-MSA.
    This version 2.0 aligns exactly with the thesis manuscript and fixes logical inconsistencies.
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
            
            # 1. Class 'X' (Noise/Silence) based on voicing threshold
            mean_voicing = float(np.mean(voicing)) if voicing.size > 0 else 0.0
            
            if mean_voicing < self.min_voicing_thresh:
                label = "Silence" # Maps to state X
                confidence = 1.0 - mean_voicing
                descriptor = {"mean_voicing": mean_voicing, "reason": "Low voicing"}
            else:
                # Analyze the last 20% of the segment (tail)
                tail_len = max(1, int(len(pitch) * 0.2))
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
                
                # Corrected logic for A vs C matching the thesis manuscript:
                # - A (Antecedent): slope > slope_epsilon OR tail_energy > energy_tau
                # - C (Consequent): slope < -slope_epsilon AND tail_energy < (energy_tau / 2.0)
                # - Fallback: If neither condition is met, use sign of the slope.
                
                if slope > self.slope_epsilon or tail_energy > self.energy_tau:
                    label = "Antecedent"
                    descriptor = {"f0_slope": slope, "energy_tail": tail_energy}
                elif slope < -self.slope_epsilon and tail_energy < (self.energy_tau / 2.0):
                    label = "Consequent"
                    descriptor = {"f0_slope": slope, "energy_tail": tail_energy}
                else:
                    # Fallback/Ambiguous cases
                    if slope < 0:
                        label = "Consequent"
                    else:
                        label = "Antecedent"
                    descriptor = {"f0_slope": slope, "energy_tail": tail_energy, "fallback": True}
                
                confidence = 0.8 # Fixed confidence for logic-based labels

            annotations.append(
                MelodySegmentAnnotation(
                    segment=segment,
                    label=label,
                    confidence=confidence,
                    descriptor=descriptor,
                )
            )

        return annotations
