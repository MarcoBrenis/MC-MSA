"""Segment classification utilities (v1.1, rule-based from MIR literature)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .features import MelodyFeatures
from .segmenter import MelodySegment


@dataclass
class MelodySegmentAnnotation:
    """Annotated segment with descriptive statistics and label."""

    segment: MelodySegment
    label: str
    confidence: float
    descriptor: dict


def _safe_polyfit(times: np.ndarray, values: np.ndarray) -> float:
    """Compute slope with a fall-back for short segments."""
    if times.size < 2:
        return 0.0
    times = times - times[0]
    slope = np.polyfit(times, values, 1)[0]
    return float(slope)


class MelodyClassifier:
    """
    Classifies melody segments using contour + tension heuristics inspiradas en:

    - LBDM (Cambouropoulos): fuerza de frontera en cambios de intervalo.
    - Modelos de pregunta–respuesta (Goldenberg).
    - Features locales de cadencia (Bigo et al.).
    - Tensión paramétrica (Farbood): altura + dinámica.
    - Arch melódico típico de frases occidentales (melodic arch).
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
        label_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        # Pendiente mínima para considerar claramente asc/desc
        self.slope_threshold = slope_threshold
        # Rango mínimo para considerar desarrollo / contornos amplios
        self.range_threshold = range_threshold
        self.range_high = range_high
        # Umbrales de altura relativa al promedio global
        self.end_high_threshold = end_high_threshold
        self.end_low_threshold = end_low_threshold
        # Umbrales de tensión normalizada
        self.tension_high = tension_high
        self.tension_low = tension_low
        # Aumento mínimo de tensión para marcar transición
        self.transition_delta_tension = transition_delta_tension
        # Permite renombrar etiquetas (ej. "pregunta" -> "Q", "respuesta" -> "A")
        self.label_aliases = (
            {k.lower(): v for k, v in label_aliases.items()}
            if label_aliases
            else {}
        )

    def _segment_descriptor(
        self,
        features: MelodyFeatures,
        segment: MelodySegment,
        global_pitch_mean: float,
        global_pitch_std: float,
        global_energy_mean: float,
        global_energy_std: float,
    ) -> dict:
        """Compute descriptor vector for one segment."""

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

        # Relative position en el registro global
        end_rel = float((pitch[-1] - global_pitch_mean) / (global_pitch_std + 1e-6))
        start_rel = float((pitch[0] - global_pitch_mean) / (global_pitch_std + 1e-6))

        # Melodic arch: pico interno vs extremos
        peak_idx = int(np.argmax(pitch))
        center_rel = float(
            (pitch[peak_idx] - global_pitch_mean) / (global_pitch_std + 1e-6)
        )

        # Duration temporal del segmento
        duration = float(times[-1] - times[0]) if times.size > 1 else 0.0

        # Tension paramétrica (pitch alto + energía alta)
        pitch_z = (pitch - global_pitch_mean) / (global_pitch_std + 1e-6)
        energy_z = (energy - global_energy_mean) / (global_energy_std + 1e-6)
        tension_mean = float(0.5 * np.mean(pitch_z) + 0.5 * np.mean(energy_z))

        descriptor = {
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
        return descriptor

    def _classify_descriptor(
        self, descriptor: dict, index: int, total: int
    ) -> str:
        """
        Aplica reglas heurísticas para decidir la etiqueta funcional.
        El orden de los if refleja prioridad teórica (cadencias > Q/A > desarrollo > transición > afirmación).
        """
        slope = descriptor["slope"]
        pitch_range = descriptor["pitch_range"]
        end_rel = descriptor["end_rel"]
        start_rel = descriptor["start_rel"]
        center_rel = descriptor["center_rel"]
        energy_delta = descriptor["energy_delta"]
        tension = descriptor["tension_mean"]
        prev_tension = descriptor.get("prev_tension", tension)

        slope_abs = abs(slope)

        # 1) CADENCE: final closure, descenso / estabilización + caída de energía/tensión
        if index == total - 1:
            if (
                end_rel < self.end_low_threshold
                and slope <= 0.0
                and energy_delta <= 0.0
            ):
                return "cadencia"

            # Cierre más neutro pero aún frase-final
            if slope_abs < self.slope_threshold and end_rel < 0.0:
                return "cadencia"

        # 2) EXPOSITION: first segment, presentación estable del material
        if index == 0:
            if (
                slope_abs < self.slope_threshold
                and pitch_range <= self.range_threshold
                and self.tension_low <= tension <= self.tension_high
            ):
                return "exposicion"

        # 3) QUESTION: contorno ascendente / final alto, tensión creciente
        if (
            (slope > self.slope_threshold or descriptor["delta_pitch"] > self.range_threshold)
            and end_rel > self.end_high_threshold
            and tension >= prev_tension
        ):
            return "pregunta"

        # 4) ANSWER: contorno descendente / regreso al registro medio/bajo
        if (
            (slope < -self.slope_threshold or descriptor["delta_pitch"] < -self.range_threshold)
            and (end_rel < start_rel or end_rel < 0.0)
        ):
            return "respuesta"

        # 5) DEVELOPMENT: interior, rango amplio y alta tensión (pico del arch melódico)
        if 0 < index < total - 1:
            if (
                pitch_range >= self.range_high
                and tension >= self.tension_high
                and center_rel > 0.0
            ):
                return "desarrollo"

        # 6) TRANSITION: aumento claro de tensión pero sin llegar a pregunta
        if (
            tension - prev_tension >= self.transition_delta_tension
            and slope > 0.0
        ):
            return "transicion"

        # 7) AFIRMACIÓN: fallback neutro
        return "afirmacion"

    def classify(
        self, features: MelodyFeatures, segments: List[MelodySegment]
    ) -> List[MelodySegmentAnnotation]:
        annotations: List[MelodySegmentAnnotation] = []
        if not segments:
            return annotations

        # Estadísticos globales para normalizar (registro y energía)
        global_pitch = features.pitch_midi
        global_energy = features.energy

        global_pitch_mean = float(np.mean(global_pitch))
        global_pitch_std = float(np.std(global_pitch) + 1e-6)
        global_energy_mean = float(np.mean(global_energy))
        global_energy_std = float(np.std(global_energy) + 1e-6)

        prev_tension = 0.0
        for i, segment in enumerate(segments):
            descriptor = self._segment_descriptor(
                features,
                segment,
                global_pitch_mean,
                global_pitch_std,
                global_energy_mean,
                global_energy_std,
            )

            # Guardamos tensión anterior para reglas de transición/pregunta
            descriptor["prev_tension"] = prev_tension
            prev_tension = descriptor["tension_mean"]

            label = self._classify_descriptor(descriptor, i, len(segments))

            # Confianza sencilla basada en qué tan fuerte es la pendiente
            slope = descriptor["slope"]
            confidence = float(
                1.0 - min(abs(slope) / (self.slope_threshold + 1e-6), 1.0)
            )

            if self.label_aliases:
                label = self.label_aliases.get(label.lower(), label)

            annotations.append(
                MelodySegmentAnnotation(
                    segment=segment,
                    label=label,
                    confidence=confidence,
                    descriptor=descriptor,
                )
            )
        return annotations


__all__ = ["MelodySegmentAnnotation", "MelodyClassifier"]
