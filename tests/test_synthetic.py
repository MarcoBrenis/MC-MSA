import numpy as np
import pytest
from matplotlib import pyplot as plt

from melody_analysis.classifier import MelodyClassifier
from melody_analysis.features import MelodyFeatures
from melody_analysis.pipeline import MelodyAnalyzer
from melody_analysis.segmenter import MelodySegmenter
from melody_analysis.visualization import (
    plot_melody_contour,
    plot_spectrogram_with_segments,
)


def _synthetic_features():
    times = np.linspace(0, 12, 121)

    pitch = np.concatenate(
        [
            np.full(40, 60.0),
            np.linspace(60.0, 65.0, 20),
            np.linspace(65.0, 63.0, 10),
            np.linspace(63.0, 61.0, 10),
            np.linspace(61.0, 60.0, 10),
            np.full(31, 59.0),
        ]
    )
    energy = np.concatenate(
        [
            np.full(40, 1.2),
            np.linspace(1.0, 1.25, 20),
            np.linspace(1.25, 1.1, 10),
            np.linspace(1.1, 0.9, 10),
            np.linspace(0.9, 0.8, 10),
            np.linspace(0.8, 0.6, 31),
        ]
    )

    energy = energy / energy.max()
    confidence = np.ones_like(times)
    return MelodyFeatures(times=times, pitch_midi=pitch, confidence=confidence, energy=energy)


def test_segmenter_detects_major_changes():
    features = _synthetic_features()
    segmenter = MelodySegmenter(kernel_size=1, peak_threshold=0.15, min_separation=6)
    segments = segmenter.segment(features)

    # We expect the melody to be broken into at least four phrases
    assert len(segments) >= 4
    # Ensure the first and last boundaries align with the data range
    assert segments[0].start_time == pytest.approx(0.0, abs=0.2)
    assert segments[-1].end_time == pytest.approx(features.times[-1], abs=0.2)


def test_classifier_assigns_melodic_roles():
    features = _synthetic_features()
    segmenter = MelodySegmenter(kernel_size=1, peak_threshold=0.15, min_separation=6)
    classifier = MelodyClassifier(slope_threshold=0.3, range_threshold=0.8)

    segments = segmenter.segment(features)
    annotations = classifier.classify(features, segments)

    labels = [ann.label for ann in annotations]
    assert labels[0] in {"exposicion", "afirmacion"}
    assert any(l in {"pregunta", "respuesta", "afirmacion"} for l in labels)
    assert labels[-1] in {"cadencia", "respuesta", "afirmacion"}


def test_analyzer_works_with_precomputed_features():
    features = _synthetic_features()
    analyzer = MelodyAnalyzer()
    result = analyzer.analyze_features(features)

    assert result.segments
    assert sum(seg.segment.duration() for seg in result.segments) > 0
    assert any(seg.label in {"pregunta", "respuesta", "afirmacion", "exposicion"} for seg in result.segments)


def test_visualizations_generate_images(tmp_path):
    features = _synthetic_features()
    analyzer = MelodyAnalyzer()
    result = analyzer.analyze_features(features)

    melody_path = tmp_path / "melodia.png"
    contour_fig = plot_melody_contour(result, output_path=melody_path)
    contour_fig.clf()
    plt.close(contour_fig)
    assert melody_path.exists()

    sample_rate = 22050
    duration = float(features.times[-1])
    t = np.linspace(0.0, duration, int(duration * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t)

    sections_path = tmp_path / "secciones.png"
    spec_fig = plot_spectrogram_with_segments(
        audio,
        sample_rate,
        result,
        output_path=sections_path,
    )
    spec_fig.clf()
    plt.close(spec_fig)
    assert sections_path.exists()
