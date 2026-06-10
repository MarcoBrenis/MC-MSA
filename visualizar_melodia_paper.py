"""Analysis of a single melody using the CLEI Paper Classifier (A/C/X logic)."""

from pathlib import Path
import sys
import soundfile as sf
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodyClassifierPaper,
    plot_melody_contour,
    plot_melody_only,
    plot_energy_only,
    plot_melody_and_energy,
    plot_spectrogram_with_segments,
    synthesize_melody,
)
from src.melody_analysis_v2.visualization import (
    plot_self_similarity,
    plot_boundary_detection,
)

def main() -> None:
    script_dir = Path(__file__).parent.absolute()
    
    # Selection of audio file
    audio_files = list(script_dir.glob("*.mp3")) + list(script_dir.glob("*.wav"))
    if not audio_files:
        print("No .mp3 or .wav files found in directory.")
        return
        
    print("\nAvailable files:")
    for i, f in enumerate(audio_files):
        print(f"{i+1}. {f.name}")
    
    file_idx = input(f"Select file [1-{len(audio_files)}] (default 1): ").strip()
    try:
        audio_path = audio_files[int(file_idx)-1] if file_idx else audio_files[0]
    except (ValueError, IndexError):
        audio_path = audio_files[0]
 
    # Analysis Configuration
    print("\nChoose melody extraction method:")
    print("1. pyin (Probabilistic YIN)")
    print("2. crepe (Deep Learning)")
    print("3. ensemble (pyin + crepe)")
    print("4. melodia (Essentia)")
    print("5. demucs_crepe (Separation + CREPE)")
    print("6. bs_roformer_rmvpe (BS-RoFormer + RMVPE)")
    print("7. rmvpe (Only RMVPE)")
    
    opcion = input("Option [1-7] (default 7): ").strip()
    metodos = {
        "1": "pyin", "2": "crepe", "3": "ensemble",
        "4": "melodia", "5": "demucs_crepe", 
        "6": "bs_roformer_rmvpe", "7": "rmvpe"
    }
    metodo_elegido = metodos.get(opcion, "rmvpe")
 
    print(f"\nAnalyzing {audio_path.name} with {metodo_elegido}...")
    
    # Initialize with the PAPER classifier
    paper_classifier = MelodyClassifierPaper()
    analyzer = MelodyAnalyzer(extraction_method=metodo_elegido, classifier=paper_classifier)
 
    try:
        resultado = analyzer.analyze_file(str(audio_path))
    except Exception as exc:
        print(f"Error during analysis: {exc}")
        sys.exit(1)
 
    print("\nClassified segments (CLEI Paper - Q/A/Silence):")
    for segmento in resultado.segments:
        print(f"{segmento.label:>5} | {segmento.segment.start_time:7.3f} → {segmento.segment.end_time:7.3f} s | {segmento.descriptor}")
 
    # Visualization
    output_dir = script_dir / "salidas_paper" / audio_path.stem / metodo_elegido
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Contour
    contour_fig = plot_melody_contour(resultado)
    contour_fig.savefig(output_dir / "contour_paper.png", dpi=150)
    plt.close(contour_fig)

    # Contour only
    contour_only_fig = plot_melody_only(resultado, show_segments=False)
    contour_only_fig.savefig(output_dir / "contour_only_paper.png", dpi=150)
    plt.close(contour_only_fig)

    # Energy only
    energy_only_fig = plot_energy_only(resultado)
    energy_only_fig.savefig(output_dir / "energy_only_paper.png", dpi=150)
    plt.close(energy_only_fig)

    # Contour and Energy
    contour_and_energy_fig = plot_melody_and_energy(resultado)
    contour_and_energy_fig.savefig(output_dir / "contour_and_energy_paper.png", dpi=150)
    plt.close(contour_and_energy_fig)
 
    # Segmented Spectrogram
    y, sr = sf.read(str(audio_path))
    if y.ndim > 1: y = np.mean(y, axis=1)
    sections_fig = plot_spectrogram_with_segments(y, sr, resultado)
    sections_fig.savefig(output_dir / "segmented_spectrogram_paper.png", dpi=150)
    plt.close(sections_fig)
 
    print(f"\nResults saved in: {output_dir}")

if __name__ == "__main__":
    main()
