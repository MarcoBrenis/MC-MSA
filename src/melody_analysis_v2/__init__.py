"""Clon experimental del paquete de análisis y segmentación melódica."""

from .features import MelodyFeatures, extract_melody_features
from .segmenter import MelodySegment, MelodySegmenter
from .segmenter_beta import MelodySegmenterBeta
from .classifier import MelodySegmentAnnotation, MelodyClassifier
from .classifier_paper import MelodyClassifierPaper
from .classifier_paper_v2 import MelodyClassifierPaperV2
from .classifier_v1_rules import MelodyClassifierV1Rules
from .pipeline import MelodyAnalyzer, MelodyAnalysisResult, analyze_melody
from .synthesis import synthesize_melody
from .visualization import (
    plot_boundary_detection,
    plot_descriptor_summary,
    plot_energy_only,
    plot_f0_only,
    plot_f0_no_segments,
    plot_melody_and_energy,
    plot_melody_contour,
    plot_energy_contour,
    plot_melody_only,
    plot_segment_extraction,
    plot_self_similarity,
    plot_spectrogram_with_segments,
)
from .exporter import DiagramExporter, export_diagram_images

__all__ = [
    "MelodyFeatures",
    "MelodySegment",
    "MelodySegmenter",
    "MelodySegmenterBeta",
    "MelodySegmentAnnotation",
    "MelodyClassifier",
    "MelodyClassifierPaper",
    "MelodyClassifierPaperV2",
    "MelodyClassifierV1Rules",
    "MelodyAnalyzer",
    "MelodyAnalysisResult",
    "extract_melody_features",
    "analyze_melody",
    "synthesize_melody",
    "plot_boundary_detection",
    "plot_descriptor_summary",
    "plot_energy_only",
    "plot_f0_no_segments",
    "plot_f0_only",
    "plot_melody_and_energy",
    "plot_melody_only",
    "plot_melody_contour",
    "plot_energy_contour",
    "plot_segment_extraction",
    "plot_self_similarity",
    "plot_spectrogram_with_segments",
    "DiagramExporter",
    "export_diagram_images",
]
