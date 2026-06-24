"""Visualization utilities for the experimental version of the analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import os
import sys

import matplotlib


def _configure_backend() -> None:
    """Choose a Matplotlib backend compatible with the environment."""

    requested = os.environ.get("MPLBACKEND")
    if requested:
        try:
            matplotlib.use(requested)
        except Exception:
            pass
        else:
            return

    current = matplotlib.get_backend().lower()
    if "agg" not in current:
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

    matplotlib.use("Agg")


_configure_backend()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:  # pragma: no cover
    import librosa
    import librosa.display  # noqa: F401
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .pipeline import MelodyAnalysisResult

# --- English Translation and Metadata Mapping for Thesis/Paper ---
CAPLIN_METADATA = {
    "presentation": {"abbr": "Pres", "full": "Presentation", "color": "#A0E0E6"},
    "continuation": {"abbr": "Cont", "full": "Continuation", "color": "#F4C2E0"},
    "antecedent": {"abbr": "A", "full": "Antecedent", "color": "#EF9A9A"},
    "consequent": {"abbr": "C", "full": "Consequent", "color": "#A5D6A7"},
    "cadential extension": {"abbr": "CE", "full": "Cadential Extension", "color": "#CDB5B1"},
    "silence": {"abbr": "S", "full": "Silence", "color": "#E0E0E0"},
    "question": {"abbr": "Q", "full": "Question", "color": "#EF9A9A"},
    "answer": {"abbr": "A", "full": "Answer", "color": "#A5D6A7"},
    "cadence": {"abbr": "Cad", "full": "Cadence", "color": "#CDB5B1"},
    "exposition": {"abbr": "Exp", "full": "Exposition", "color": "#A0E0E6"},
    "development": {"abbr": "Dev", "full": "Development", "color": "#F4C2E0"},
    "transition": {"abbr": "Trans", "full": "Transition", "color": "#FFF59D"},
    "statement": {"abbr": "Stmt", "full": "Statement", "color": "#D1C4E9"},
}

def _get_caplin_meta(label: str) -> dict:
    """Retrieves abbreviation, full name and color for a given label, supporting dynamic translation."""
    label_lower = label.lower().strip()
    
    # Mapping various forms to normalized English keys
    mapping = {
        "presentación": "presentation",
        "presentacion": "presentation",
        "presentation": "presentation",
        "p": "presentation",
        
        "continuación": "continuation",
        "continuacion": "continuation",
        "continuation": "continuation",
        "c": "continuation",
        
        "antecedente": "antecedent",
        "antecedent": "antecedent",
        "a": "antecedent",
        
        "consecuente": "consequent",
        "consequent": "consequent",
        "cns": "consequent",
        "cons": "consequent",
        
        "extensión cadencial": "cadential extension",
        "extension cadencial": "cadential extension",
        "cadential extension": "cadential extension",
        "ec": "cadential extension",
        "ce": "cadential extension",
        
        "silencio": "silence",
        "silence": "silence",
        "rest": "silence",
        "s": "silence",
        "x": "silence",
        
        "pregunta": "question",
        "question": "question",
        "q": "question",
        
        "respuesta": "answer",
        "answer": "answer",
        "r": "answer",
        "resp": "answer",
        
        "cadencia": "cadence",
        "cadence": "cadence",
        "cad": "cadence",
        
        "exposición": "exposition",
        "exposicion": "exposition",
        "exposition": "exposition",
        "exp": "exposition",
        
        "desarrollo": "development",
        "development": "development",
        "dev": "development",
        "des": "development",
        
        "transición": "transition",
        "transicion": "transition",
        "transition": "transition",
        "tra": "transition",
        "trans": "transition",
        
        "afirmación": "statement",
        "afirmacion": "statement",
        "affirmation": "statement",
        "statement": "statement",
        "afi": "statement",
        "stmt": "statement"
    }
    
    normalized_key = mapping.get(label_lower, label_lower)
    return CAPLIN_METADATA.get(normalized_key, {
        "abbr": label[:3].upper(),
        "full": label.title(),
        "color": "tab:gray"
    })

def _label_color(label: str) -> str:
    return _get_caplin_meta(label)["color"]

def _format_label(label: str) -> str:
    """Returns the English abbreviation for in-graph display."""
    return _get_caplin_meta(label)["abbr"]


def _get_plot_step(n_points: int, target: int = 5000) -> int:
    """Determines the downsampling step for plotting to avoid system hangs."""
    if n_points <= target:
        return 1
    return int(np.ceil(n_points / target))


def _midi_to_hz(pitch_midi: np.ndarray) -> np.ndarray:
    """Convert MIDI values to fundamental frequency (f0) in Hz."""

    pitch_midi = np.asarray(pitch_midi, dtype=float)
    return 440.0 * np.power(2.0, (pitch_midi - 69.0) / 12.0)


def _ensure_output_path(output_path: Optional[Path]) -> Optional[Path]:
    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _draw_segment_overlays(ax: plt.Axes, segments: Iterable, ymax: float) -> None:
    import matplotlib.patches as mpatches
    
    legend_handles = {}
    segments_list = list(segments)
    
    total_duration = max(ann.segment.end_time for ann in segments_list) if segments_list else 1.0
    min_duration = total_duration * 0.02
    
    for ann in segments_list:
        label = ann.label
        meta = _get_caplin_meta(label)
        
        display_abbr = meta["abbr"]
        display_full = meta["full"]
        color = meta["color"]
        
        # Color span
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.25,
        )
        
        # Track handle for legend
        if display_abbr not in legend_handles:
            # We format the label as "Abbr: Full Name" to mimic the tabular request
            legend_label = f"{display_abbr}: {display_full}"
            legend_handles[display_abbr] = mpatches.Patch(color=color, alpha=0.5, label=legend_label)

        # Label on top (Abbreviated)
        duration = ann.segment.end_time - ann.segment.start_time
        # Only show text if segment is wide enough to avoid overlap
        if duration > min_duration: 
            ax.text(
                (ann.segment.start_time + ann.segment.end_time) / 2,
                ymax,
                display_abbr,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight='bold',
                color="black",
                alpha=0.9
            )

    if legend_handles:
        # Sort handles by the desired order in CAPLIN_METADATA, avoiding duplicates
        planned_order = []
        for v in CAPLIN_METADATA.values():
            abbr = v["abbr"]
            if abbr not in planned_order:
                planned_order.append(abbr)
        sorted_abbrs = [a for a in planned_order if a in legend_handles]
        # Add any unexpected abbreviations at the end
        for a in legend_handles:
            if a not in sorted_abbrs:
                sorted_abbrs.append(a)
                
        handles = [legend_handles[abbr] for abbr in sorted_abbrs]
        
        # Create legend outside the plot area
        ax.legend(
            handles=handles, 
            title="Formal Functions", 
            loc='upper left', 
            bbox_to_anchor=(1.05, 1), 
            fontsize='x-small',
            frameon=True,
            title_fontsize='small'
        )


def plot_melody_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    show_segments: bool = False,
    title: Optional[str] = None,
) -> Figure:
    """Plot only the melody pitch, with optional segments."""

    times = result.features.times
    pitch = result.features.pitch_midi
    f0_hz = _midi_to_hz(pitch)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, pitch, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (MIDI)")

    ax_hz = ax.twinx()
    ax_hz.plot(times, f0_hz, color="tab:red", alpha=0.5)
    ax_hz.set_ylabel("f0 (Hz)", color="tab:red")
    ax_hz.tick_params(axis="y", labelcolor="tab:red")

    if show_segments:
        ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    if title is None:
        ax.set_title("f0 Contour")
    else:
        ax.set_title(title)
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
    title: Optional[str] = None,
) -> Figure:
    """Plot only the f0 curve in Hz."""

    times = result.features.times
    f0_hz = _midi_to_hz(result.features.pitch_midi)

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(times, f0_hz, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f0 (Hz)")

    if show_segments:
        ymax = float(np.nanmax(f0_hz)) if f0_hz.size else 0.0
        _draw_segment_overlays(ax, result.segments, ymax)

    if title is None:
        ax.set_title("f0 Contour")
    else:
        ax.set_title(title)
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
    title: Optional[str] = None,
) -> Figure:
    """Plot the f0 curve in Hz without showing segments."""

    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy
    f0_hz = _midi_to_hz(pitch)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(times, energy, color="tab:green", alpha=0.6)
    ax2.set_ylabel("Normalized energy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    ax3 = ax1.twinx()
    ax3.plot(times, f0_hz, color="tab:red", alpha=0.5)
    ax3.set_ylabel("f0 (Hz)", color="tab:red")
    ax3.tick_params(axis="y", labelcolor="tab:red")

    if title is None:
        ax1.set_title("f0 Contour and normalized energy")
    else:
        ax1.set_title(title)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_energy_only(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    title: Optional[str] = None,
) -> Figure:
    """Plot only the normalized energy curve."""
    times = result.features.times
    energy = result.features.energy

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, energy, color="tab:green", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Energy")
    if title is None:
        ax.set_title("Normalized Energy")
    else:
        ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_melody_and_energy(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    title: Optional[str] = None,
) -> Figure:
    """Plot the f0 contour and normalized energy without segments."""
    times = result.features.times
    pitch = result.features.pitch_midi
    energy = result.features.energy

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(times, pitch, color="tab:blue", label="Pitch (MIDI)", linewidth=1.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(times, energy, color="tab:green", alpha=0.6, label="Normalized energy", linewidth=1.2)
    ax2.set_ylabel("Normalized energy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    if title is None:
        ax1.set_title("f0 Contour and normalized energy")
    else:
        ax1.set_title(title)
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
    title: Optional[str] = None,
) -> Figure:
    times = result.features.times
    pitch = result.features.pitch_midi

    # Adaptive downsampling for visualization performance
    step = _get_plot_step(len(times))
    
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(times[::step], pitch[::step], color="tab:blue", linewidth=1.0)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (MIDI)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ymax = float(np.nanmax(pitch)) if pitch.size else 0.0
    _draw_segment_overlays(ax1, result.segments, ymax)

    if title is None:
        ax1.set_title("f0 Contour and detected segments")
    else:
        ax1.set_title(title)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_energy_contour(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    title: Optional[str] = None,
) -> Figure:
    """Plot only the normalized energy with segment overlays."""
    times = result.features.times
    energy = result.features.energy

    # Adaptive downsampling for visualization performance
    step = _get_plot_step(len(times))
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times[::step], energy[::step], color="tab:green", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Energy", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")

    _draw_segment_overlays(ax, result.segments, 1.0)

    if title is None:
        ax.set_title("Normalized Energy and detected segments")
    else:
        ax.set_title(title)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def _infer_hop_length(times: np.ndarray, sample_rate: int) -> int:
    if times.size < 2:
        return 512
    dt = float(np.median(np.diff(times)))
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
    title: Optional[str] = None,
) -> Figure:
    if librosa is None:
        raise ImportError("librosa is required to generate spectrograms")

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
    fig.colorbar(img, ax=ax, format="%.0f dB", label="Intensity")

    f0_hz = _midi_to_hz(result.features.pitch_midi)
    f0_mel = librosa.hz_to_mel(f0_hz)
    
    # Plot f0 with adaptive downsampling
    step = _get_plot_step(len(f0_mel))
    ax.plot(result.features.times[::step], f0_mel[::step], color="white", linewidth=1.5, alpha=0.9, label="f0")

    ymax = S_db.shape[0]
    total_duration = max(ann.segment.end_time for ann in result.segments) if result.segments else 1.0
    min_duration = total_duration * 0.02
    
    for ann in result.segments:
        label = ann.label
        display_label = _format_label(label)
        color = _label_color(label)
        
        ax.axvspan(
            ann.segment.start_time,
            ann.segment.end_time,
            color=color,
            alpha=0.2,
            linewidth=0,
        )
        duration = ann.segment.end_time - ann.segment.start_time
        if duration > min_duration:
            ax.text(
                (ann.segment.start_time + ann.segment.end_time) / 2,
                ymax - 1,
                display_label,
                ha="center",
                va="top",
                color="black",
                fontsize=7,
                bbox={"facecolor": color, "alpha": 0.4, "pad": 1},
            )

    if title is None:
        ax.set_title("Spectrogram with annotated segments and f0")
    else:
        ax.set_title(title)
    ax.legend(loc="upper right")
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
    cmap: str = "viridis",
    title: Optional[str] = None,
) -> Figure:
    """Show the self-similarity matrix used during segmentation."""

    if result.self_similarity is None:
        raise ValueError("No self-similarity matrix found in result.")

    ssm = np.asarray(result.self_similarity)
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(ssm, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Frame index")
    if title is None:
        ax.set_title("Self-similarity matrix")
    else:
        ax.set_title(title)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    fig.colorbar(img, ax=ax, label="Similarity")
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_boundary_detection(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    title: Optional[str] = None,
) -> Figure:
    """Plot novelty curves used to detect boundaries."""

    if result.novelty is None and result.base_novelty is None and result.ssm_novelty is None:
        raise ValueError("No novelty curves available in result.")

    times = result.features.times
    def _time_axis(target: np.ndarray) -> np.ndarray:
        if times.size:
            return np.linspace(times[0], times[-1], num=len(target))
        return np.arange(len(target))

    fig, ax = plt.subplots(figsize=(10, 3))
    if result.base_novelty is not None:
        ax.plot(_time_axis(result.base_novelty), result.base_novelty, label="Δ + energy", color="tab:blue")
    if result.ssm_novelty is not None:
        ax.plot(_time_axis(result.ssm_novelty), result.ssm_novelty, label="Self-similarity", color="tab:orange")
    if result.novelty is not None:
        ax.plot(_time_axis(result.novelty), result.novelty, label="Combined", color="tab:red", linewidth=2)

    ax.set_xlabel("Time (s)" if times.size else "Frame")
    ax.set_ylabel("Novelty")
    if title is None:
        ax.set_title("Boundary detection (novelty curves)")
    else:
        ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_segment_extraction(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    title: Optional[str] = None,
) -> Figure:
    """Visualize only the detected segment bands."""

    if not result.segments:
        raise ValueError("No segments to show.")

    fig, ax = plt.subplots(figsize=(10, 1.6))
    if result.features.times.size:
        ax.set_xlim(result.features.times[0], result.features.times[-1])
    ax.set_ylim(0, 1)
    _draw_segment_overlays(ax, result.segments, ymax=0.9)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time (s)")
    if title is None:
        ax.set_title("Segment extraction")
    else:
        ax.set_title(title)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_descriptor_summary(
    result: MelodyAnalysisResult,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    metrics: Optional[Iterable[str]] = None,
) -> Figure:
    """Summarize descriptors per segment in separate plots."""

    if not result.segments:
        raise ValueError("No segment annotations available.")

    default_metrics = ("slope", "pitch_range", "energy_mean", "tension_mean")
    metrics = tuple(metrics) if metrics is not None else default_metrics

    names = [f"seg{i}" for i in range(len(result.segments))]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 2.5 * len(metrics)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, key in zip(axes, metrics):
        values = [ann.descriptor.get(key, 0.0) for ann in result.segments]
        colors = [_label_color(ann.label) for ann in result.segments]
        ax.bar(names, values, color=colors)
        ax.set_ylabel(key)
        ax.grid(True, axis="y", alpha=0.2)
    axes[-1].set_xlabel("Segment")
    fig.suptitle("Descriptor computation per segment", y=0.99)
    fig.tight_layout()

    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_melspectrogram(
    audio: np.ndarray,
    sample_rate: int,
    *,
    output_path: Optional[Path] = None,
    dpi: int = 300,
    cmap: str = "magma",
    title: Optional[str] = None,
) -> Figure:
    """Plot a clean Mel-spectrogram in English."""
    if librosa is None:
        raise ImportError("librosa is required to generate spectrograms")
        
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax, cmap=cmap)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Power (dB)")
    if title is None:
        ax.set_title("Mel-spectrogram")
    else:
        ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    
    output_path = _ensure_output_path(output_path)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        
    return fig


__all__ = [
    "plot_f0_no_segments",
    "plot_f0_only",
    "plot_melody_only",
    "plot_energy_only",
    "plot_melody_and_energy",
    "plot_melody_contour",
    "plot_energy_contour",
    "plot_self_similarity",
    "plot_boundary_detection",
    "plot_segment_extraction",
    "plot_descriptor_summary",
    "plot_spectrogram_with_segments",
    "plot_melspectrogram",
]
