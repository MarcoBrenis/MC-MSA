"""Utility functions to visualizar resultados del análisis melódico."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import os
import sys

import matplotlib


def _configure_backend() -> None:
    """Escoger un backend de Matplotlib compatible con el entorno."""

    requested = os.environ.get("MPLBACKEND")
    if requested:
        # El usuario ha indicado explícitamente el backend: lo respetamos.
        try:
            matplotlib.use(requested)
        except Exception:
            # Si la selección manual falla, continuamos con la detección automática.
            pass
        else:
            return

    current = matplotlib.get_backend().lower()
    if "agg" not in current:
        # Ya tenemos un backend interactivo o inline (por ejemplo, Jupyter).
        return

    display_available = any(
        os.environ.get(var)
        for var in ("DISPLAY", "WAYLAND_DISPLAY", "MPLBACKEND")
    ) or sys.platform == "darwin" or sys.platform.startswith("win")

    if display_available:
        for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "MacOSX"):
            try:
                matplotlib.use(candidate)
            except Exception:
                continue
            else:
                return

    # Si nada funcionó, mantenemos Agg para garantizar que se puedan guardar archivos.
    matplotlib.use("Agg")
_configure_backend()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:  # pragma: no cover - dependencia opcional para espectrogramas
    import librosa
    import librosa.display  # noqa: F401 - activa utilidades de visualización
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .pipeline import MelodyAnalysisResult
from .segmenter import MelodySegmenter

LABEL_COLOR_MAP = {
    "exposicion": "tab:blue",
    "desarrollo": "tab:orange",
    "pregunta": "tab:red",
    "respuesta": "tab:green",
    "transicion": "tab:purple",
    "cadencia": "tab:brown",
    "afirmacion": "tab:gray",
}

LABEL_ALIAS_COLOR_MAP = {
    "q": LABEL_COLOR_MAP["pregunta"],
    "a": LABEL_COLOR_MAP["respuesta"],
}


def _label_color(label: str) -> str:
    """Asignar un color consistente a cada etiqueta funcional."""

    label_lower = label.lower()
    if label_lower in LABEL_COLOR_MAP:
        return LABEL_COLOR_MAP[label_lower]
    if label_lower in LABEL_ALIAS_COLOR_MAP:
        return LABEL_ALIAS_COLOR_MAP[label_lower]
    return "tab:gray"


def _midi_to_hz(pitch_midi: np.ndarray) -> np.ndarray:
    """Convertir valores MIDI a frecuencia fundamental (f0) en Hz."""

    pitch_midi = np.asarray(pitch_midi, dtype=float)
    return 440.0 * np.power(2.0, (pitch_midi - 69.0) / 12.0)


def _ensure_output_path(output_path: Optional[Path]) -> Optional[Path]:
    """Crear la carpeta de salida si se especifica una ruta."""

    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _draw_segment_overlays(ax: plt.Axes, segments: Iterable, ymax: float) -> None:
    """Sombrear cada segmento en el eje temporal."""

    for ann in segments:
        color = _label_color(ann.label)
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.15,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax,
            ann.label,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
            color=color,
        )


def plot_melody_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = True,
) -> Figure:
    """Graficar únicamente el contorno melódico opcionalmente con segmentos."""

    times = result.features.times
    pitch = result.features.pitch_midi
    f0_hz = _midi_to_hz(pitch)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, pitch, label="Melodía (MIDI)", color="tab:blue")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Pitch (MIDI)")

    # Secondary axis for f0 in Hz
    ax_hz = ax.twinx()
    ax_hz.plot(times, f0_hz, label="f0 (Hz)", color="tab:red", alpha=0.5)
    ax_hz.set_ylabel("f0 (Hz)", color="tab:red")
    ax_hz.tick_params(axis="y", labelcolor="tab:red")

    if show_segments:
        ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    ax.set_title("Contorno melódico")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_f0_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = False,
) -> Figure:
    """Graficar únicamente la curva de f0 en Hz sin el contorno en MIDI."""

    times = result.features.times
    f0_hz = _midi_to_hz(result.features.pitch_midi)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, f0_hz, label="f0 (Hz)", color="tab:red")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("f0 (Hz)")

    if show_segments:
        ymax = float(np.nanmax(f0_hz)) if f0_hz.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    ax.set_title("Curva de frecuencia fundamental (solo f0)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_f0_no_segments(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Graficar solo la curva de f0 en Hz sin resaltar segmentos."""

    times = result.features.times
    f0_hz = _midi_to_hz(result.features.pitch_midi)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, f0_hz, label="f0 (Hz)", color="tab:red")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("f0 (Hz)")
    ax.set_title("Frecuencia fundamental (sin segmentos)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_signal_and_novelty(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = True,
) -> Figure:
    """Mostrar la señal normalizada junto a las curvas de novedad.

    Se grafica la señal de audio normalizada ("smoothed" por librosa al
    centrarse en ventanas) y, debajo, la novedad base suavizada y la novedad
    combinada con autosimilitud si está disponible.
    """

    if result.normalized_audio is None or result.sample_rate is None:
        raise ValueError(
            "El resultado no contiene audio normalizado; usa analyze_audio/analyze_file para obtenerlo."
        )

    audio = result.normalized_audio
    sr = result.sample_rate
    audio_times = np.linspace(0.0, len(audio) / sr, num=len(audio))

    novelty = result.novelty
    base_novelty = result.base_novelty
    ssm_novelty = result.ssm_novelty

    fig, (ax_sig, ax_nov) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax_sig.plot(audio_times, audio, color="tab:gray", linewidth=0.8)
    ax_sig.set_title("Señal normalizada")
    ax_sig.set_xlabel("Tiempo (s)")
    ax_sig.set_ylabel("Amplitud")
    ax_sig.grid(True, alpha=0.2)

    times = result.features.times
    if base_novelty is not None:
        ax_nov.plot(times, base_novelty, label="Novedad derivativa (suavizada)", color="tab:blue")
    if ssm_novelty is not None:
        ax_nov.plot(times, ssm_novelty, label="Novedad por SSM", color="tab:orange", alpha=0.8)
    if novelty is not None:
        ax_nov.plot(times, novelty, label="Novedad combinada", color="tab:red", linewidth=1.5)

    if show_segments:
        ymax = 1.05 * float(np.nanmax(novelty)) if novelty is not None else 1.0
        _draw_segment_overlays(ax_nov, result.segments, ymax)

    ax_nov.set_title("Curvas de novedad")
    ax_nov.set_xlabel("Tiempo (s)")
    ax_nov.set_ylabel("Intensidad relativa")
    ax_nov.grid(True, alpha=0.2)
    ax_nov.legend()

    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_self_similarity(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Graficar la matriz de autosimilitud usada por el segmentador."""

    sim = result.self_similarity
    if sim is None:
        sim = MelodySegmenter().compute_self_similarity(result.features)

    fig, ax = plt.subplots(figsize=(5, 4))
    img = ax.imshow(sim, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Matriz de autosimilitud (pitch + energía)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    fig.colorbar(img, ax=ax, label="Similitud")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_melody_contour(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
) -> Figure:
    """Generar un gráfico del contorno melódico con los segmentos resaltados."""

    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy
    f0_hz = _midi_to_hz(pitch)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, label="Melodía (MIDI)", color="tab:blue")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
    _draw_segment_overlays(ax1, result.segments, ymax)

    ax2 = ax1.twinx()
    ax2.plot(times, energy, label="Energía", color="tab:green", alpha=0.6)
    ax2.set_ylabel("Energía normalizada", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    # Show f0 in Hz on a third axis to avoid overlaps
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.plot(times, f0_hz, label="f0 (Hz)", color="tab:red", alpha=0.5)
    ax3.set_ylabel("f0 (Hz)", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Contorno melódico y segmentos detectados")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def _infer_hop_length(times: np.ndarray, sample_rate: int) -> int:
    """Inferir el hop length a partir de la cuadrícula temporal."""

    if times.size < 2:
        return 512
    delta = np.diff(times)
    dt = float(np.median(delta))
    hop = int(np.round(dt * sample_rate))
    return max(hop, 1)


def plot_spectrogram_with_segments(
    audio: np.ndarray,
    sample_rate: int,
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    cmap: str = "magma",
) -> Figure:
    """Dibujar un espectrograma mel con los segmentos del JSON sobrepuestos."""

    if librosa is None:
        raise ImportError("librosa es necesario para generar espectrogramas")

    hop_length = _infer_hop_length(result.features.times, sample_rate)
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=2048,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap=cmap,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%.0f dB", label="Intensidad")

    # Plot f0 over the mel axis to locate the fundamental
    f0_hz = _midi_to_hz(result.features.pitch_midi)
    f0_mel = librosa.hz_to_mel(f0_hz)
    ax.plot(result.features.times, f0_mel, color="white", linewidth=1.5, alpha=0.9, label="f0")

    ymax = S_db.shape[0]
    for ann in result.segments:
        color = _label_color(ann.label)
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.text(
            (ann.segment.start_time + ann.segment.end_time) / 2,
            ymax - 1,
            ann.label,
            ha="center",
            va="top",
            color="black",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.35, "pad": 1},
        )

    ax.set_title("Espectrograma mel con secciones anotadas y f0")
    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


__all__ = [
    "plot_f0_no_segments",
    "plot_f0_only",
    "plot_melody_only",
    "plot_melody_contour",
    "plot_spectrogram_with_segments",
]
