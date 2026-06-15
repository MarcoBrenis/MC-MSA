from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    plot_f0_no_segments,
    plot_f0_only,
    plot_melody_only,
    plot_melody_contour,
    plot_spectrogram_with_segments,
    plot_energy_only,
    plot_melody_and_energy,
)
import librosa

analyzer = MelodyAnalyzer()
resultado = analyzer.analyze_file("1.mp3")
for segmento in resultado.segments:
    print(segmento.label, segmento.segment.start_time, segmento.segment.end_time)
    

# Generate visualizations directly from Python
fig1 = plot_melody_contour(resultado)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
# If you only want the melodic contour without energy or f0:
fig_melodia = plot_melody_only(resultado)
# If you only want the f0 curve in Hz:
fig_f0 = plot_f0_only(resultado)
# If you prefer f0 without any segment overlays:
fig_f0_plano = plot_f0_no_segments(resultado)
# Only energy:
fig_energia = plot_energy_only(resultado)
# Contour y energía:
fig_melodia_energia = plot_melody_and_energy(resultado)


"""Commented example to analyze a melody and generate visualizations."""

# --- Importing Libraries ---
from pathlib import Path  # To handle file paths when saving images.

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
    MelodyClassifier,  # Allows defining aliases like Q/A.
    plot_boundary_detection,  # Novelty curves to detect boundaries.
    plot_descriptor_summary,  # Summary of descriptors per segment.
    plot_melody_contour,  # Function to plot the melodic contour.
    plot_segment_extraction,  # Only the detected segment bands.
    plot_self_similarity,  # Self-similarity matrix of the analyzer.
    plot_spectrogram_with_segments,  # Function to plot the spectrogram with sections.
)


def main() -> None:
    """Executes analysis on a file and shows the results."""

    # --- Melody Analysis ---
    # Define path to audio file to analyze.
    audio_path = "1.mp3"
    # If you want aliases (e.g. "antecedent"→"A" and "consequent"→"C"),
    # replace the following line with the commented one below.
    analyzer = MelodyAnalyzer()
    # analyzer = MelodyAnalyzer(
    #     classifier=MelodyClassifier(label_aliases={"antecedent": "A", "consequent": "C"})
    # )
    # Call method to analyze file, which returns an object with results.
    resultado = analyzer.analyze_file(audio_path)

    # Print summary of detected segments to console.
    print("Detected segments:")
    # Iterate over each segment in results to show main data.
    for segmento in resultado.segments:
        # Print label, start time, and end time with three decimals.
        print(f"{segmento.label:>12} | {segmento.segment.start_time:7.3f} → {segmento.segment.end_time:7.3f} s")

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
    output_dir = Path("salidas_visualizacion")
    output_dir.mkdir(exist_ok=True)

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
    print(f"Contour figure saved at: {contour_path.resolve()}")

    # --- Manual Mel-spectrogram Visualization ---
    # Load audio file with soundfile to access signal (y) and sampling rate (sr).
    y, sr = sf.read(audio_path)
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
    print(f"Segmented Mel-spectrogram saved at: {sections_path.resolve()}")

    # --- Visualizations per Stage (Separated) ---
    if resultado.self_similarity is not None:
        ssm_fig = plot_self_similarity(resultado)
        ssm_path = output_dir / "autosimilitud.png"
        ssm_fig.savefig(ssm_path, dpi=150, bbox_inches="tight")
        if not interactive_backend:
            plt.close(ssm_fig)
        print(f"Self-similarity matrix saved at: {ssm_path.resolve()}")
    else:
        print("No self-similarity matrix available in the result.")

    try:
        novelty_fig = plot_boundary_detection(resultado)
    except ValueError:
        novelty_fig = None
        print("No novelty curves to plot boundary detection.")
    else:
        novelty_path = output_dir / "boundary_detection.png"
        novelty_fig.savefig(novelty_path, dpi=150, bbox_inches="tight")
        if not interactive_backend:
            plt.close(novelty_fig)
        print(f"Boundary detection saved at: {novelty_path.resolve()}")

    segs_fig = plot_segment_extraction(resultado)
    segs_path = output_dir / "segment_extraction.png"
    segs_fig.savefig(segs_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(segs_fig)
    print(f"Segment extraction saved at: {segs_path.resolve()}")

    desc_fig = plot_descriptor_summary(resultado)
    desc_path = output_dir / "descriptor_summary.png"
    desc_fig.savefig(desc_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(desc_fig)
    print(f"Descriptor summary saved at: {desc_path.resolve()}")

    # Si hay backend interactivo se realiza un plt.show() final para abrir las ventanas.
    if interactive_backend:
        print("Opening Matplotlib windows...")
        plt.show()


if __name__ == "__main__":
    main()
