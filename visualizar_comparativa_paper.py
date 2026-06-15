"""Comparison of original and cover melodies using LCS similarity and Paper Classifier."""

from pathlib import Path
import sys
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodyClassifierPaper,
    plot_melody_contour,
    plot_segment_extraction,
    plot_spectrogram_with_segments,
)
from src.melody_analysis_v2.visualization import (
    plot_self_similarity,
    plot_boundary_detection,
)
from src.melody_analysis_v2.classifier_paper import calculate_lcs

def save_plots(resultado, audio_path, output_dir):
    """Generates and saves all plots for a given analysis result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Melodic contour (with segments)
    try:
        fig = plot_melody_contour(resultado)
        fig.savefig(output_dir / "melodic_contour.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting contour for {audio_path.name}: {e}")

    # 1b. Segment extraction (bands only)
    try:
        fig = plot_segment_extraction(resultado)
        fig.savefig(output_dir / "segment_extraction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting segment bands for {audio_path.name}: {e}")

    # 2. SSM Matrix
    try:
        fig = plot_self_similarity(resultado)
        fig.savefig(output_dir / "self_similarity_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting SSM for {audio_path.name}: {e}")

    # 3. Boundary detection
    try:
        fig = plot_boundary_detection(resultado)
        fig.savefig(output_dir / "boundary_detection.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting boundaries for {audio_path.name}: {e}")

    # 4. Segmented spectrogram
    try:
        y, sr = sf.read(str(audio_path))
        if y.ndim > 1: y = np.mean(y, axis=1)
        fig = plot_spectrogram_with_segments(y, sr, resultado)
        fig.savefig(output_dir / "segmented_spectrogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting spectrogram for {audio_path.name}: {e}")

def main() -> None:
    script_dir = Path(__file__).parent.absolute()
    
    print("--- Melody Comparison (Original vs Cover) ---")
    
    # Selection of files
    audio_files = list(script_dir.glob("*.mp3")) + list(script_dir.glob("*.wav"))
    if len(audio_files) < 2:
        print("At least 2 audio files are required for comparison.")
        return
        
    print("\nAvailable files:")
    for i, f in enumerate(audio_files):
        print(f"{i+1}. {f.name}")
    
    idx1 = input(f"Select ORIGINAL file [1-{len(audio_files)}]: ").strip()
    idx2 = input(f"Select COVER file [1-{len(audio_files)}]: ").strip()
    
    try:
        path_orig = audio_files[int(idx1)-1]
        path_cover = audio_files[int(idx2)-1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    # Method selection
    print("\nExtraction method:")
    print("1. pyin (Probabilistic YIN - Fast)")
    print("2. crepe (Deep Learning - Accurate)")
    print("3. ensemble (Combination - Robust)")
    print("4. melodia (Essentia - Standard)")
    print("5. demucs_crepe (Separation + CREPE)")
    print("6. bs_roformer_rmvpe (BS-RoFormer + RMVPE)")
    print("7. rmvpe (Only RMVPE)")
    
    op = input("Option [1-7] (default 7): ").strip()
    metodos = {
        "1": "pyin", "2": "crepe", "3": "ensemble",
        "4": "melodia", "5": "demucs_crepe", 
        "6": "bs_roformer_rmvpe", "7": "rmvpe"
    }
    method = metodos.get(op, "rmvpe")

    classifier = MelodyClassifierPaper()
    analyzer = MelodyAnalyzer(extraction_method=method, classifier=classifier)

    # Output directory for plots
    comp_dir = script_dir / "salidas_comparativa" / f"{path_orig.stem}_vs_{path_cover.stem}" / method
    
    print(f"\nAnalyzing Original: {path_orig.name}...")
    res_orig = analyzer.analyze_file(str(path_orig))
    save_plots(res_orig, path_orig, comp_dir / "original")
    
    print(f"Analyzing Cover: {path_cover.name}...")
    res_cover = analyzer.analyze_file(str(path_cover))
    save_plots(res_cover, path_cover, comp_dir / "cover")

    # Extract sequences
    seq_orig = [s.label for s in res_orig.segments]
    seq_cover = [s.label for s in res_cover.segments]

    # Map labels to English initials for sequence display
    label_to_initial = {
        "pregunta": "Q",
        "question": "Q",
        "q": "Q",
        "respuesta": "A",
        "answer": "A",
        "r": "A",
        "resp": "A",
        "silencio": "S",
        "silence": "S",
        "s": "S",
        "x": "S",
        "rest": "S",
        "antecedente": "A",
        "antecedent": "A",
        "consecuente": "C",
        "consequent": "C",
    }
    
    seq_orig_mapped = [label_to_initial.get(s.lower(), s) for s in seq_orig]
    seq_cover_mapped = [label_to_initial.get(s.lower(), s) for s in seq_cover]

    # Calculate LCS
    similarity = calculate_lcs(seq_orig, seq_cover)

    # Display academic table
    print("\n" + "="*60)
    print(f"{'MELODIC STRUCTURE COMPARATIVE TABLE':^60}")
    print("="*60)
    print(f"{'Song':<20} | {'Tag Sequence (Formal Functions)':<35}")
    print("-"*60)
    print(f"{'Original':<20} | {'-'.join(seq_orig_mapped):<35}")
    print(f"{'Cover':<20} | {'-'.join(seq_cover_mapped):<35}")
    print("-"*60)
    print(f"{'LCS Similarity:':<20} | {similarity:4.2%}")
    print("="*60)
    print("\nLegend: Q (Question), A (Answer), S (Rest/Silence)")

    # Save report
    report_path = comp_dir / f"comparison_{path_orig.stem}_vs_{path_cover.stem}.txt"
    with open(report_path, "w") as f:
        f.write(f"Comparison: {path_orig.name} vs {path_cover.name}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Original: {'-'.join(seq_orig_mapped)}\n")
        f.write(f"Cover:    {'-'.join(seq_cover_mapped)}\n")
        f.write(f"LCS Similarity: {similarity:.4f}\n")
    
    print(f"\nReport and plots saved in: {comp_dir}")

if __name__ == "__main__":
    main()
