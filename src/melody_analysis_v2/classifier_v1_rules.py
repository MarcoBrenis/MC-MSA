"""Clasificador basado en las reglas originales de MC-MSA (v1.x)."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment
from .classifier import MelodySegmentAnnotation


def _safe_polyfit(times: np.ndarray, values: np.ndarray) -> float:
    """Calcula la pendiente de forma segura."""
    if times.size < 2:
        return 0.0
    times = times - times[0]
    slope = np.polyfit(times, values, 1)[0]
    return float(slope)


class MelodyClassifierV1Rules:
    """
    Clasificador que replica la lógica heurística de la versión 1.x (MC-MSA).
    Usa umbrales de pendiente, rango melódico y tensión.
    """

    def __init__(
        self,
        *,
        slope_threshold: float = 0.4,
        range_threshold: float = 3.0,
        range_high: float = 5.0,
        end_high_threshold: float = 1.0,
        end_low_threshold: float = -1.0,
        tension_high: float = 0.8,
        tension_low: float = -0.2,
        transition_delta_tension: float = 0.25,
    ) -> None:
        self.slope_threshold = slope_threshold
        self.range_threshold = range_threshold
        self.range_high = range_high
        self.end_high_threshold = end_high_threshold
        self.end_low_threshold = end_low_threshold
        self.tension_high = tension_high
        self.tension_low = tension_low
        self.transition_delta_tension = transition_delta_tension

    def _segment_descriptor(
        self,
        features: MelodyFeatures,
        segment: MelodySegment,
        global_pitch_mean: float,
        global_pitch_std: float,
        global_energy_mean: float,
        global_energy_std: float,
    ) -> dict:
        idx = slice(segment.start_index, segment.end_index + 1)
        times = features.times[idx]
        pitch = features.pitch_midi[idx]
        energy = features.energy[idx]

        # Contour
        slope = _safe_polyfit(times, pitch)
        delta_pitch = float(pitch[-1] - pitch[0])
        pitch_range = float(np.max(pitch) - np.min(pitch))

        # Energy
        energy_mean = float(np.mean(energy))
        energy_delta = float(energy[-1] - energy[0])

        # Relative position
        end_rel = float((pitch[-1] - global_pitch_mean) / (global_pitch_std + 1e-6))
        start_rel = float((pitch[0] - global_pitch_mean) / (global_pitch_std + 1e-6))

        # Melodic arch
        peak_idx = int(np.argmax(pitch))
        center_rel = float((pitch[peak_idx] - global_pitch_mean) / (global_pitch_std + 1e-6))

        # Duration
        duration = float(times[-1] - times[0]) if times.size > 1 else 0.0

        # Tension
        pitch_z = (pitch - global_pitch_mean) / (global_pitch_std + 1e-6)
        energy_z = (energy - global_energy_mean) / (global_energy_std + 1e-6)
        tension_mean = float(0.5 * np.mean(pitch_z) + 0.5 * np.mean(energy_z))

        return {
            "slope": slope,
            "delta_pitch": delta_pitch,
            "pitch_range": pitch_range,
            "energy_mean": energy_mean,
            "energy_delta": energy_delta,
            "start_rel": start_rel,
            "end_rel": end_rel,
            "center_rel": center_rel,
            "duration": duration,
            "tension_mean": tension_mean,
        }

    def _classify_descriptor(self, descriptor: dict, index: int, total: int) -> str:
        slope = descriptor["slope"]
        pitch_range = descriptor["pitch_range"]
        end_rel = descriptor["end_rel"]
        start_rel = descriptor["start_rel"]
        center_rel = descriptor["center_rel"]
        energy_delta = descriptor["energy_delta"]
        tension = descriptor["tension_mean"]
        prev_tension = descriptor.get("prev_tension", tension)

        slope_abs = abs(slope)

        # 1) CADENCE: final closure
        if index == total - 1:
            if end_rel < self.end_low_threshold and slope <= 0.0 and energy_delta <= 0.0:
                return "Cadencia"
            if slope_abs < self.slope_threshold and end_rel < 0.0:
                return "Cadencia"

        # 2) EXPOSITION: first segment
        if index == 0:
            if (slope_abs < self.slope_threshold and 
                pitch_range <= self.range_threshold and 
                self.tension_low <= tension <= self.tension_high):
                return "Exposición"

        # 3) QUESTION
        if ((slope > self.slope_threshold or descriptor["delta_pitch"] > self.range_threshold) and 
            end_rel > self.end_high_threshold and tension >= prev_tension):
            return "Pregunta"

        # 4) ANSWER
        if ((slope < -self.slope_threshold or descriptor["delta_pitch"] < -self.range_threshold) and 
            (end_rel < start_rel or end_rel < 0.0)):
            return "Respuesta"

        # 5) DEVELOPMENT
        if 0 < index < total - 1:
            if (pitch_range >= self.range_high and 
                tension >= self.tension_high and 
                center_rel > 0.0):
                return "Desarrollo"

        # 6) TRANSITION
        if (tension - prev_tension >= self.transition_delta_tension and slope > 0.0):
            return "Transición"

        return "Afirmación"

    def classify(
        self, features: MelodyFeatures, segments: List[MelodySegment], sim_matrix: Optional[np.ndarray] = None, ssm_step: int = 1
    ) -> List[MelodySegmentAnnotation]:
        annotations: List[MelodySegmentAnnotation] = []
        if not segments:
            return annotations

        global_pitch = features.pitch_midi
        global_energy = features.energy

        global_pitch_mean = float(np.mean(global_pitch))
        global_pitch_std = float(np.std(global_pitch) + 1e-6)
        global_energy_mean = float(np.mean(global_energy))
        global_energy_std = float(np.std(global_energy) + 1e-6)

        prev_tension = 0.0
        for i, segment in enumerate(segments):
            # Silence handling (voicing mask)
            idx = slice(segment.start_index, segment.end_index + 1)
            voicing = features.confidence[idx]
            if np.mean(voicing) < 0.1: # Silence detected by voicing
                annotations.append(MelodySegmentAnnotation(segment=segment, label="Silencio", confidence=1.0, descriptor={"is_silence": True}))
                continue

            descriptor = self._segment_descriptor(
                features, segment, global_pitch_mean, global_pitch_std, global_energy_mean, global_energy_std
            )
            descriptor["prev_tension"] = prev_tension
            prev_tension = descriptor["tension_mean"]

            label = self._classify_descriptor(descriptor, i, len(segments))
            slope = descriptor["slope"]
            confidence = float(1.0 - min(abs(slope) / (self.slope_threshold + 1e-6), 1.0))

            annotations.append(MelodySegmentAnnotation(segment=segment, label=label, confidence=confidence, descriptor=descriptor))
            
        return annotations
