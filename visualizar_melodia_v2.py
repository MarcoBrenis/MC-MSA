"""Commented example to analyze a melody and generate visualizations."""

# --- Importing Libraries ---
from pathlib import Path  # To handle file paths when saving images.
import sys

# Necessary tools for analysis and visualization are imported.
import soundfile as sf  # To read audio files.
import librosa  # For complementary audio analysis routines.
import librosa.display  # To represent the spectrogram on a time-frequency axis.
import matplotlib  # To query which backend is being used.
import matplotlib.pyplot as plt  # To create and show plots.
import numpy as np  # For numerical operations, especially with arrays.

# Import the main class and visualization helpers of clone v2.
from src.melody_analysis_v2 import (
    MelodyAnalyzer,  # Encapsulates extraction, segmentation, and classification.
    MelodyClassifierPaper,  # New classifier for the paper.
    plot_melody_contour,  # Function to plot the melodic contour.
    plot_melody_only,  # Function to plot only the melodic contour without segments or energy.
    plot_energy_only,  # Function to plot only the energy.
    plot_melody_and_energy,  # Function to plot contour and energy.
    plot_spectrogram_with_segments,  # Function to plot the spectrogram with sections.
    synthesize_melody,  # Function to synthesize the extracted melody.
)
from src.melody_analysis_v2.visualization import (
    plot_self_similarity,
    plot_boundary_detection,
)


def main() -> None:
    """Executes analysis on a file and shows the results."""

    # --- Melody Analysis ---
    # Define path to audio file to analyze.
    # We use a path relative to the script for robustness.
    script_dir = Path(__file__).parent.absolute()
    audio_path = script_dir / "1.mp3"
    
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return

    # --- Analysis Configuration ---
    print("Choose melody extraction method:")
    print("1. pyin (Probabilistic YIN - Fast)")
    print("2. crepe (Deep Learning - Accurate)")
    print("3. ensemble (Combination of both - Robust)")
    print("4. melodia (Essentia - Classic standard for polyphony)")
    print("5. demucs_crepe (Voice separation + CREPE)")
    print("6. bs_roformer_rmvpe (BS-RoFormer + RMVPE)")
    print("7. rmvpe (Only RMVPE)")
    opcion = input("Option [1-7] (default 1): ").strip()
    
    metodos = {
        "1": "pyin", 
        "2": "crepe", 
        "3": "ensemble",
        "4": "melodia",
        "5": "demucs_crepe",
        "6": "bs_roformer_rmvpe",
        "7": "rmvpe"
    }
    metodo_elegido = metodos.get(opcion, "pyin")
    
    # --- Classifier Configuration ---
    print("\nChoose classifier:")
    print("1. Standard (Full Caplin: Antecedent, Consequent, Presentation, etc.)")
    print("2. CLEI Paper (Strict A, C, X)")
    op_clf = input("Option [1/2] (default 1): ").strip()
    
    if op_clf == "2":
        classifier = MelodyClassifierPaper()
        print("Using Paper classifier (A/C/X).")
    else:
        classifier = None # Will use the default MelodyClassifier
        print("Using Standard classifier.")

    # Create an instance of the melody analyzer with chosen method and classifier.
    analyzer = MelodyAnalyzer(extraction_method=metodo_elegido, classifier=classifier)
    # Call method to analyze file, which returns an object with results.
    try:
        resultado = analyzer.analyze_file(str(audio_path))
    except ImportError as exc:
        print(f"\nCould not run method '{metodo_elegido}': {exc}")
        if metodo_elegido == "crepe":
            print(
                "\nSuggestions:\n"
                "- Use option 1 ('pyin') to continue working without TensorFlow.\n"
                "- If you need CREPE, create a virtual environment with a Python version "
                "compatible with TensorFlow and reinstall dependencies."
            )
        elif metodo_elegido == "ensemble":
            print(
                "\nSuggestion:\n"
                "- 'ensemble' also depends on CREPE. If TensorFlow is not available, "
                "use option 1 ('pyin')."
            )
        elif metodo_elegido == "demucs_crepe":
            print(
                "\nSuggestion:\n"
                "- 'demucs_crepe' also ends up using CREPE. If TensorFlow is not "
                "available, try first with 'pyin' or 'melodia'."
            )
        sys.exit(1)

    # Print summary of detected segments to console.
    print("Formally classified segments (Caplin Rules):")
    # Iterate over each segment in results to show main data.
    for segmento in resultado.segments:
        # Extract SSM similarity score if it exists, or 0.0 if first segment
        sim_score = segmento.descriptor.get("ssm_similarity_with_previous", 0.0)
        # Print label, similarity score, and times
        print(f"{segmento.label:>20} (SSM Sim: {sim_score:4.2f}) | {segmento.segment.start_time:7.3f} → {segmento.segment.end_time:7.3f} s")

    # Query active Matplotlib backend to decide if windows will be shown.
    backend = matplotlib.get_backend()
    backend_lower = backend.lower()
    interactive_backend = not backend_lower.endswith("agg")

    if interactive_backend:
        print(f"Matplotlib backend is '{backend}', figures will be shown at the end.")
    else:
        print(
            "Non-interactive backend (Agg u otro similar): figures will be saved to disk.\n"
            "To open interactive windows you can define MPLBACKEND=TkAgg (u otro backend)\n"
            "before running the script, provided you have dependencies installed."
        )

    # --- Melodic Contour Visualization ---
    # Ensure an output directory to save generated figures.
    # Now we organize by extraction method.
    output_base_dir = script_dir / "salidas_visualizacion"
    output_dir = output_base_dir / metodo_elegido
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a figure with the melodic contour using the helper included in the package.
    contour_fig = plot_melody_contour(resultado)
    # Define the path where the melodic contour image will be saved.
    contour_path = output_dir / "contorno_melodico.png"
    # Save figure to disk with good resolution.
    contour_fig.savefig(contour_path, dpi=150, bbox_inches="tight")
    # If no interactive backend is available, close the figure to free resources.
    if not interactive_backend:
        plt.close(contour_fig)
    # Inform in console where the image was stored.
    print(f"Contour figure saved at: {contour_path}")

    # Get a figure of only melodic contour without segments or energy.
    melody_only_fig = plot_melody_only(resultado, show_segments=False)
    # Define path where image will be saved.
    melody_only_path = output_dir / "contorno_melodico_solo.png"
    melody_only_fig.savefig(melody_only_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(melody_only_fig)
    print(f"Figure of only contour saved at: {melody_only_path}")

    # Get a figure of only the energy.
    energy_only_fig = plot_energy_only(resultado)
    energy_only_path = output_dir / "energia_solo.png"
    energy_only_fig.savefig(energy_only_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(energy_only_fig)
    print(f"Figure of only energy saved at: {energy_only_path}")

    # Get a figure of melodic contour and energy.
    melody_energy_fig = plot_melody_and_energy(resultado)
    melody_energy_path = output_dir / "contorno_y_energia.png"
    melody_energy_fig.savefig(melody_energy_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(melody_energy_fig)
    print(f"Figure of contour and energy saved at: {melody_energy_path}")

    # --- Novelty and SSM Matrices ---
    try:
        ssm_fig = plot_self_similarity(resultado)
        ssm_path = output_dir / "matriz_autosimilitud.png"
        ssm_fig.savefig(ssm_path, dpi=150, bbox_inches="tight")
        if not interactive_backend:
            plt.close(ssm_fig)
        print(f"Self-similarity matrix saved at: {ssm_path}")
    except Exception as e:
        print(f"Could not plot SSM matrix: {e}")

    try:
        bound_fig = plot_boundary_detection(resultado)
        bound_path = output_dir / "deteccion_fronteras.png"
        bound_fig.savefig(bound_path, dpi=150, bbox_inches="tight")
        if not interactive_backend:
            plt.close(bound_fig)
        print(f"Boundary detection curves saved at: {bound_path}")
    except Exception as e:
        print(f"Could not plot boundary detection: {e}")

    # --- Manual Mel-spectrogram Visualization ---
    # Load audio file with soundfile to access signal (y) and sampling rate (sr).
    y, sr = sf.read(str(audio_path))
    # If audio is stereo (more than one channel), convert to mono by averaging channels.
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Calculate Mel-spectrogram; represents energy by perceptual bands over time.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # Convert power to decibels to highlight fine details.
    Sdb = librosa.power_to_db(S, ref=np.max)

    # Create spectrogram plot in Matplotlib and get the resulting figure.
    manual_fig, ax = plt.subplots(figsize=(14, 4))
    # Show spectrogram in plot, with time and frequency axes in Mel scale.
    librosa.display.specshow(Sdb, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    # Add a descriptive title to the plot.
    ax.set_title("Mel-spectrogram (manual)")
    # Adjust layout to avoid element overlapping.
    manual_fig.tight_layout()
    # Save manual figure to contrast it later with annotated version.
    manual_path = output_dir / "mel_espectrograma_manual.png"
    manual_fig.savefig(manual_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(manual_fig)
    # Se informa la ruta de la figura manual.
    print(f"Manual Mel-spectrogram saved at: {manual_path.resolve()}")

    # --- Automated Visualization with Sections ---
    # Generate a figure that reuses detected segments to highlight each musical block.
    sections_fig = plot_spectrogram_with_segments(y, sr, resultado)
    # Save figure with embedded sections and labels.
    sections_path = output_dir / "segmented_mel_spectrogram.png"
    sections_fig.savefig(sections_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(sections_fig)
    # Se muestra la ruta donde quedó guardado el espectrograma con secciones.
    print(f"Segmented Mel-spectrograms saved at: {sections_path.resolve()}")

    # --- Melodic Audio Export ---
    try:
        print("Synthesizing melodic audio...")
        # Synthesize audio signal from extracted features.
        audio_melodia = synthesize_melody(
            resultado.features.times,
            resultado.features.pitch_midi,
            resultado.features.confidence,
            resultado.features.energy,
            sample_rate=sr
        )
        # Define output path for the audio file.
        audio_output_path = output_dir / "melodia_extraida.wav"
        # Save audio file using soundfile.
        sf.write(str(audio_output_path), audio_melodia, sr)
        print(f"Melodic audio saved at: {audio_output_path.resolve()}")
    except Exception as e:
        print(f"Error exporting melodic audio: {e}")

    # Si hay backend interactivo se realiza un plt.show() final para abrir las ventanas.
    if interactive_backend:
        print("Opening Matplotlib windows...")
        plt.show()


if __name__ == "__main__":
    main()
