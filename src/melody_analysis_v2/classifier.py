"""Segment classification utilities (Caplin formal functions using SSM and internal features)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment


@dataclass
class MelodySegmentAnnotation:
    """Annotated segment with descriptive statistics and Caplin label."""

    segment: MelodySegment
    label: str
    confidence: float
    descriptor: dict


class MelodyClassifier:
    """
    Classifies melody segments using William Caplin's formal structural functions.
    
    This classifier relies on the structural repetition found in the 
    Self-Similarity Matrix (SSM) and internal acoustic features (slope, energy).
    """

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.65,
    ) -> None:
        """
        Parameters
        ----------
        similarity_threshold:
            The minimum cross-correlation or distance score in the SSM required
            to consider a segment a structural repetition of another (Basic Idea -> Presentation).
        """
        self.similarity_threshold = similarity_threshold
        self.tonic_pc: Optional[int] = None  # Pitch Class of the tonic (0-11)

    def _detect_tonic(self, pitch_midi: np.ndarray) -> int:
        """Detects the song tonic using a histogram of pitch classes."""
        # Filter out NaNs and convert to integers
        valid_pitches = pitch_midi[~np.isnan(pitch_midi)]
        if valid_pitches.size == 0:
            return 0  # Default to C
            
        # Fold to octave (chroma / pitch class)
        chromas = np.round(valid_pitches).astype(int) % 12
        
        # Count occurrences of each pitch class
        counts = np.bincount(chromas, minlength=12)
        
        # The most frequent pitch class is our tonic candidate
        return int(np.argmax(counts))

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

    def _extract_descriptors(self, features: MelodyFeatures, segment: MelodySegment, global_pitch_mean: float) -> dict:
        """Calculates internal acoustic properties of a single musical segment."""
        idx = slice(segment.start_index, segment.end_index + 1)
        pitch = features.pitch_midi[idx]
        energy = features.energy[idx]
        times = features.times[idx]
        confidence = features.confidence[idx]
        
        # Identify "voiced" frames within the segment
        voiced_mask = confidence > 0.1
        voiced_pitch = pitch[voiced_mask]
        voiced_times = times[voiced_mask]
        
        # Pitch Slope (Ascending tension vs Descending resolution)
        # Use only voiced frames for slope computation
        slope = self._safe_polyfit(voiced_times, voiced_pitch, 1) if voiced_times.size > 1 else 0.0
        
        # Pitch Range (Wide range often indicates Continuation fragmentation)
        valid_pitch = voiced_pitch[~np.isnan(voiced_pitch)]
        pitch_range = float(np.max(valid_pitch) - np.min(valid_pitch)) if valid_pitch.size > 0 else 0.0
        
        # Pitch Relative to Global Tonic (Does it end high or low?)
        pitch_mean = float(np.mean(valid_pitch)) if valid_pitch.size > 0 else 0.0
        pitch_end = float(valid_pitch[-1]) if valid_pitch.size > 0 else pitch_mean
        end_rel = pitch_end - global_pitch_mean # positive means it hangs high (Half Cadence), negative is low (Perfect)
        
        # Energy Delta (Significant drop indicates Cadence closure)
        energy_mean = float(np.mean(energy)) if energy.size > 0 else 0.0
        energy_delta = float(energy[-1] - energy[0]) if energy.size > 0 else 0.0
        
        # Silence check: if more than 80% of frames are unvoiced
        is_silence = np.mean(voiced_mask) < 0.2
        
        return {
            "duration": segment.duration(),
            "pitch_slope": slope,
            "pitch_range": pitch_range,
            "pitch_end_relative": end_rel,
            "energy_mean": energy_mean,
            "energy_delta": energy_delta,
            "is_silence": is_silence,
            "pitch_end_absolute": pitch_end,
            "ssm_similarity_with_previous": 0.0 # Placeholder
        }

    def _get_segment_similarity(self, sim_matrix: np.ndarray, seg1: MelodySegment, seg2: MelodySegment, ssm_step: int = 1) -> float:
        """Calculates the mean similarity between two temporal segments using the SSM."""
        if sim_matrix is None or seg1.end_index <= seg1.start_index or seg2.end_index <= seg2.start_index:
            return 0.0
            
        # Scale indices for downsampled SSM
        s1_start, s1_end = seg1.start_index // ssm_step, seg1.end_index // ssm_step
        s2_start, s2_end = seg2.start_index // ssm_step, seg2.end_index // ssm_step
        
        # Ensure we don't exceed SSM bounds
        s1_end = min(s1_end, sim_matrix.shape[0])
        s2_end = min(s2_end, sim_matrix.shape[1])
        
        if s1_end <= s1_start or s2_end <= s2_start:
            return 0.0
            
        block = sim_matrix[s1_start:s1_end, s2_start:s2_end]
        return float(np.mean(block))

    def _detect_groups_of_four(self, segment_data: List[dict], sim_matrix: np.ndarray, tonic: int, ssm_step: int = 1) -> None:
        """Group segments into thematic blocks using SSM and internal heuristics."""
        
        # Filter out silence segments for thematic relationship calculation
        musical_indices = [i for i, d in enumerate(segment_data) if d["caplin_label"] != "Silence"]
        
        for idx_in_musical, i in enumerate(musical_indices[:-1]):
            s1_data = segment_data[i]
            
            # Find next musical segment
            next_i = musical_indices[idx_in_musical + 1]
            s2_data = segment_data[next_i]
            
            if s1_data["caplin_label"] != "unknown":
                continue
                
            s1 = s1_data["segment"]
            s2 = s2_data["segment"]
            sim_score = self._get_segment_similarity(sim_matrix, s1, s2, ssm_step=ssm_step)
            s2_data["descriptor"]["ssm_similarity_with_previous"] = sim_score
            
            # --- Caplin Rules with Internal Heuristics ---
            
            # 1. PRESENTATION (Sentence Structure)
            if sim_score >= self.similarity_threshold and abs(s1_data["descriptor"]["pitch_slope"]) < 5.0:
                s1_data["caplin_label"] = "Presentation"
                s2_data["caplin_label"] = "Presentation"
                
                # Check for CONTINUATION
                if idx_in_musical + 2 < len(musical_indices):
                    s3_i = musical_indices[idx_in_musical + 2]
                    s3_data = segment_data[s3_i]
                    is_fragmented = s3_data["descriptor"]["duration"] < (s1_data["descriptor"]["duration"] * 0.75)
                    is_expanded = s3_data["descriptor"]["pitch_range"] > (s1_data["descriptor"]["pitch_range"] * 1.5)
                    
                    if is_fragmented or is_expanded:
                        s3_data["caplin_label"] = "Continuation"
                    else:
                        s3_data["caplin_label"] = "Continuation"
                        
                if idx_in_musical + 3 < len(musical_indices):
                    s4_i = musical_indices[idx_in_musical + 3]
                    s4_data = segment_data[s4_i]
                    if s4_data["descriptor"]["energy_delta"] < 0 and s4_data["descriptor"]["pitch_slope"] < 0:
                        s4_data["caplin_label"] = "Cadential Extension"
                    else:
                        s4_data["caplin_label"] = "Continuation"
                    
            # 2. ANTECEDENT (Period Structure)
            else:
                s1_data["caplin_label"] = "Antecedent"
                
                # Antecedent usually ends with a Half Cadence (tension)
                # HC: Ends on a note that is NOT the tonic (usually the 5th)
                s2_last_pc = int(np.round(s2_data["descriptor"]["pitch_end_absolute"])) % 12
                is_hc = s2_last_pc != tonic
                
                s2_data["caplin_label"] = "Antecedent" # Keep general label
                
                # Check for CONSEQUENT
                if idx_in_musical + 2 < len(musical_indices):
                    segment_data[musical_indices[idx_in_musical + 2]]["caplin_label"] = "Consequent"
                    
                if idx_in_musical + 3 < len(musical_indices):
                    s4_i = musical_indices[idx_in_musical + 3]
                    s4_data = segment_data[s4_i]
                    
                    # Consequent ends with a Perfect Authentic Cadence (resolution)
                    # PAC: Ends on the TONIC with falling energy/slope
                    s4_last_pc = int(np.round(s4_data["descriptor"]["pitch_end_absolute"])) % 12
                    is_pac = (s4_last_pc == tonic) and (s4_data["descriptor"]["pitch_slope"] < 0)
                    
                    s4_data["caplin_label"] = "Consequent"
 
        # Cleanup trailing unknowns
        for data in segment_data:
            if data["caplin_label"] == "unknown":
                data["caplin_label"] = "Cadential Extension" # Standard fallback for loose ends in forms

    def classify(
        self, features: MelodyFeatures, segments: List[MelodySegment], sim_matrix: Optional[np.ndarray] = None, ssm_step: int = 1
    ) -> List[MelodySegmentAnnotation]:
        
        annotations: List[MelodySegmentAnnotation] = []
        if not segments:
            return annotations

        # Compute global pitch mean for relative tonic comparisons
        valid_global_pitch = features.pitch_midi[~np.isnan(features.pitch_midi)]
        global_pitch_mean = float(np.mean(valid_global_pitch)) if valid_global_pitch.size > 0 else 60.0

        # Pre-compute descriptors for all segments
        segment_data = []
        for segment in segments:
            desc = self._extract_descriptors(features, segment, global_pitch_mean)
            label = "Silence" if desc["is_silence"] else "unknown"
            segment_data.append({
                "segment": segment,
                "descriptor": desc,
                "caplin_label": label,
                "sim_score": 0.0
            })

        # Apply structural rules if SSM is available
        if sim_matrix is not None:
            self.tonic_pc = self._detect_tonic(features.pitch_midi)
            self._detect_groups_of_four(segment_data, sim_matrix, self.tonic_pc, ssm_step=ssm_step)
            
            for data in segment_data:
                # Simple confidence based on how far we are from the SSM threshold
                if data["caplin_label"] == "Silence":
                    confidence = 1.0
                else:
                    ssm_sim = data["descriptor"].get("ssm_similarity_with_previous", 0.0)
                    confidence = float(min(1.0, abs(ssm_sim - self.similarity_threshold) * 2.0))
                
                annotations.append(
                    MelodySegmentAnnotation(
                        segment=data["segment"],
                        label=data["caplin_label"],
                        confidence=confidence,
                        descriptor=data["descriptor"],
                    )
                )
        else:
            # Fallback if SSM didn't generate properly
            for data in segment_data:
                annotations.append(
                    MelodySegmentAnnotation(
                        segment=data["segment"],
                        label="unknown",
                        confidence=0.0,
                        descriptor=data["descriptor"],
                    )
                )

        return annotations


__all__ = ["MelodySegmentAnnotation", "MelodyClassifier"]
