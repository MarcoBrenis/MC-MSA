"""CLI clonada para la versión experimental ``melody_analysis_v2``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:  # pragma: no cover
    import librosa
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

from .pipeline import MelodyAnalyzer
from .visualization import plot_melody_contour, plot_spectrogram_with_segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment and classify melodic structure from an audio file.",
    )
    parser.add_argument("audio", type=Path, help="Ruta al archivo de audio a analizar")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional donde guardar el resultado en formato JSON.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Frecuencia de muestreo objetivo para la carga del audio.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Longitud del hop empleada para la extracción de características.",
    )
    parser.add_argument(
        "--melody-plot",
        type=Path,
        default=None,
        help="Ruta donde guardar la gráfica del contorno melódico detectado (v2).",
    )
    parser.add_argument(
        "--sections-plot",
        type=Path,
        default=None,
        help="Ruta para guardar el espectrograma con las secciones (v2).",
    )

    args = parser.parse_args()

    if librosa is None:
        raise ImportError("librosa es requerida para cargar audio desde la línea de comandos")

    audio, sr = librosa.load(str(args.audio), sr=args.sr)

    analyzer = MelodyAnalyzer(sample_rate=args.sr, hop_length=args.hop_length)
    result = analyzer.analyze_audio(audio, sr)
    payload = result.to_dict()

    if args.output is None:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.melody_plot is not None:
        plot_melody_contour(result, output_path=args.melody_plot)

    if args.sections_plot is not None:
        plot_spectrogram_with_segments(audio, sr, result, output_path=args.sections_plot)


if __name__ == "__main__":
    main()
