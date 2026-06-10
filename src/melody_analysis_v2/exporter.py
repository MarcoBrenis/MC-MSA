"""Utility to export intermediate analysis steps as images for a pipeline diagram."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .classifier import MelodyClassifier, MelodySegmentAnnotation
from .features import MelodyFeatures, extract_melody_features
from .pipeline import MelodyAnalyzer, MelodyAnalysisResult
from .segmenter import MelodySegmenter
from .visualization import _label_color, _midi_to_hz


class DiagramExporter:
    """Generates individual images for each step of the melody analysis pipeline."""

    def __init__(self, output_dir: str | Path = "diagram_export"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use a clean style for thesis diagrams
        plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14})

    def _save_fig(self, fig: Figure, name: str, dpi: int = 300):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def export_all(self, audio_path: str, method: str = "pyin"):
        """Run the full pipeline and export all 9 steps."""
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # 1. Audio representation
        self.export_audio_representation(y, sr)

        # 2. Mel-spectrogram
        self.export_mel_spectrogram(y, sr)

        # 3. F0 Estimation (Raw)
        analyzer = MelodyAnalyzer(extraction_method=method)
        result = analyzer.analyze_audio(y, sr)
        
        self.export_f0_estimation(result.features)

        # 4. Smoothing and normalization
        self.export_normalized_features(result.features)

        # 5. Self-similarity matrix
        segmenter = MelodySegmenter()
        features_ds = result.features
        if len(result.features.times) > segmenter.max_ssm_frames:
            step = int(np.ceil(len(result.features.times) / segmenter.max_ssm_frames))
            features_ds = MelodyFeatures(
                times=result.features.times[::step],
                pitch_midi=result.features.pitch_midi[::step],
                confidence=result.features.confidence[::step],
                energy=result.features.energy[::step]
            )
        ssm = segmenter.compute_self_similarity(features_ds)
        self.export_ssm(ssm)

        # 6. Segment detection
        novelty = segmenter.compute_novelty(features_ds)
        boundaries = segmenter.find_boundaries(novelty)
        self.export_segment_detection(features_ds.times, novelty, boundaries)

        # 7. Feature extraction (Descriptors)
        self.export_feature_extraction(result.segments)

        # 8. Rules-based classifier
        self.export_rules_classifier(result)

        # 9. Visualization and JSON
        self.export_final_results(result, y, sr)

    def export_audio_representation(self, y: np.ndarray, sr: int):
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="tab:blue")
        ax.set_title("Audio representation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return self._save_fig(fig, "01_audio_representation")

    def export_mel_spectrogram(self, y: np.ndarray, sr: int):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel-spectrogram")
        return self._save_fig(fig, "02_mel_spectrogram")

    def export_f0_estimation(self, features: MelodyFeatures):
        fig, ax = plt.subplots(figsize=(10, 3))
        f0_hz = _midi_to_hz(features.pitch_midi)
        ax.plot(features.times, f0_hz, color="tab:red", linewidth=1.5)
        ax.set_title("Fundamental frequency estimation")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("F0 (Hz)")
        ax.grid(True, alpha=0.3)
        return self._save_fig(fig, "03_f0_estimation")

    def export_normalized_features(self, features: MelodyFeatures):
        fig, ax = plt.subplots(figsize=(10, 4))
        
        p = features.pitch_midi
        p_norm = (p - np.mean(p)) / (np.std(p) + 1e-6)
        
        ax.plot(features.times, p_norm, label="F0 (norm)", color="tab:blue")
        ax.plot(features.times, features.energy, label="Energy (norm)", color="tab:green", alpha=0.7)
        ax.plot(features.times, features.confidence, label="Voicing (norm)", color="tab:red", alpha=0.5)
        
        ax.set_title("Smoothing and normalization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        return self._save_fig(fig, "04_smoothing_normalization")

    def export_ssm(self, ssm: np.ndarray):
        fig, ax = plt.subplots(figsize=(6, 6))
        img = ax.imshow(ssm, origin="lower", cmap="viridis", interpolation="nearest")
        fig.colorbar(img, ax=ax, label="Similarity")
        ax.set_title("Self-similarity matrix")
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Time (frames)")
        return self._save_fig(fig, "05_ssm")

    def export_segment_detection(self, times: np.ndarray, novelty: np.ndarray, boundaries: np.ndarray):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(times, novelty, color="tab:purple", label="Novelty curve")
        for b in boundaries:
            ax.axvline(times[b], color="red", linestyle="--", alpha=0.6)
        
        ax.set_title("Segment detection")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Novelty / Homogeneity")
        ax.legend()
        return self._save_fig(fig, "06_segment_detection")

    def export_feature_extraction(self, annotations: List[MelodySegmentAnnotation]):
        if not annotations:
            return None
            
        descriptors = []
        labels = []
        keys = ["slope", "pitch_range", "energy_mean", "tension_mean", "duration"]
        
        for ann in annotations:
            d = [ann.descriptor.get(k, 0) for k in keys]
            descriptors.append(d)
            labels.append(ann.label)
            
        data = np.array(descriptors).T
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(data, aspect="auto", cmap="RdYlBu_r")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        fig.colorbar(img, ax=ax, label="Z-score")
        ax.set_title("Feature extraction (Segment Descriptors)")
        fig.tight_layout()
        return self._save_fig(fig, "07_feature_extraction")

    def export_rules_classifier(self, result: MelodyAnalysisResult):
        from .visualization import plot_melody_contour
        fig = plot_melody_contour(result)
        # Find axes[0] and change title
        for ax in fig.axes:
            if ax.get_title() in ["Contorno melódico y segmentos detectados", "Melodic contour and detected segments"]:
                ax.set_title("Rules-based classifier (Caplin's rules)")
                break
        return self._save_fig(fig, "08_rules_classifier")

    def export_final_results(self, result: MelodyAnalysisResult, y: np.ndarray, sr: int):
        from .visualization import plot_spectrogram_with_segments
        fig = plot_spectrogram_with_segments(y, sr, result)
        for ax in fig.axes:
             if "Espectrograma" in ax.get_title() or "Spectrogram" in ax.get_title():
                ax.set_title("Visualization and JSON file")
                break
        
        # Robust JSON serialization handling numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                if isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return json.JSONEncoder.default(self, obj)

        json_path = self.output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=4, cls=NumpyEncoder)
            
        return self._save_fig(fig, "09_final_visualization")


def export_diagram_images(audio_path: str, output_dir: str = "diagram_export", method: str = "pyin"):
    exporter = DiagramExporter(output_dir)
    exporter.export_all(audio_path, method)
    print(f"Diagram images exported to {output_dir}")
