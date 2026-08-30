"""Comparison of original and cover melodies using LCS similarity and Thesis Classifier."""

from pathlib import Path
import sys
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodyClassifierThesis,
)
from src.melody_analysis_v2.visualization import (
    plot_melody_only,
    plot_energy_only,
    plot_melody_and_energy,
    plot_melspectrogram,
    plot_self_similarity,
    plot_boundary_detection,
    plot_segment_extraction,
    plot_melody_contour,
    plot_energy_contour,
    plot_spectrogram_with_segments,
    plot_descriptor_summary,
)
from src.melody_analysis_v2.classifier_thesis import calculate_lcs

def save_plots(resultado, audio_path, output_dir):
    """Generates and saves all 11 step-by-step plots for a given analysis result, matching analyze_single_track."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load audio for spectrograms
    try:
        y, sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception as e:
        print(f"Warning: Could not load audio for spectrogram plots: {e}")
        y, sr = None, None

    # Step 1: f0 Contour
    try:
        fig = plot_melody_only(resultado, title="Step 1: f0 Contour")
        fig.savefig(output_dir / "step1_f0_contour.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 1 for {audio_path.name}: {e}")

    # Step 2: Normalized Energy
    try:
        fig = plot_energy_only(resultado, title="Step 2: Normalized Energy")
        fig.savefig(output_dir / "step2_energy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 2 for {audio_path.name}: {e}")

    # Step 3: f0 Contour & Energy
    try:
        fig = plot_melody_and_energy(resultado, title="Step 3: f0 Contour & Energy")
        fig.savefig(output_dir / "step3_contour_energy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 3 for {audio_path.name}: {e}")

    # Step 4: Mel-Spectrogram
    if y is not None:
        try:
            fig = plot_melspectrogram(y, sr, title="Step 4: Mel-Spectrogram")
            fig.savefig(output_dir / "step4_spectrogram.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting Step 4 for {audio_path.name}: {e}")

    # Step 5: Self-Similarity Matrix (SSM)
    try:
        fig = plot_self_similarity(resultado, title="Step 5: Self-Similarity Matrix (SSM)")
        fig.savefig(output_dir / "step5_ssm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 5 for {audio_path.name}: {e}")

    # Step 6: Boundary Detection (Novelty Curves)
    try:
        fig = plot_boundary_detection(resultado, title="Step 6: Boundary Detection (Novelty Curves)")
        fig.savefig(output_dir / "step6_novelty_boundaries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 6 for {audio_path.name}: {e}")

    # Step 7: Segment Extraction & Classification Bands
    try:
        fig = plot_segment_extraction(resultado, title="Step 7: Segment Extraction & Classification Bands")
        fig.savefig(output_dir / "step7_segmentation_bands.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 7 for {audio_path.name}: {e}")

    # Step 8: Segmented f0 Contour
    try:
        fig = plot_melody_contour(resultado, title="Step 8: Segmented f0 Contour")
        fig.savefig(output_dir / "step8_contour_segmented.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 8 for {audio_path.name}: {e}")

    # Step 9: Segmented Energy Contour
    try:
        fig = plot_energy_contour(resultado, title="Step 9: Segmented Normalized Energy")
        fig.savefig(output_dir / "step9_energy_segmented.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 9 for {audio_path.name}: {e}")

    # Step 10: Mel-Spectrogram with Segments
    if y is not None:
        try:
            fig = plot_spectrogram_with_segments(y, sr, resultado, title="Step 10: Mel-Spectrogram with Annotated Segments")
            fig.savefig(output_dir / "step10_spectrogram_segmented.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting Step 10 for {audio_path.name}: {e}")

    # Step 11: Descriptor Summary
    try:
        metrics_to_show = ["pitch_slope", "pitch_range", "energy_mean", "energy_delta"]
        fig = plot_descriptor_summary(resultado, metrics=metrics_to_show)
        fig.savefig(output_dir / "step11_descriptor_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Error plotting Step 11 for {audio_path.name}: {e}")


def plot_caplin_bands(res_orig, res_cover, output_path: Path, meta_orig=None, meta_cover=None):
    from matplotlib.lines import Line2D
    from src.melody_analysis_v2.visualization import _get_caplin_meta
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    plt.subplots_adjust(hspace=0.6)
    
    legend_handles_dict = {}
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        meta = meta_orig if i == 0 else meta_cover
        if meta:
            t_title, t_artist = meta
            ax.set_title(f"Melodic Segmentation (Bands) - {name}\nSong: {t_title} | Performer: {t_artist}", fontweight='bold', fontsize=10)
        else:
            ax.set_title(f"Melodic Segmentation (Bands) - {name}", fontweight='bold')
            
        ax.set_yticks([])
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        ax.set_xlim(0, max_time)
        
        for seg in res.segments:
            start_time = seg.segment.start_time
            end_time = seg.segment.end_time
            label = seg.label
            duration = end_time - start_time
            
            meta_info = _get_caplin_meta(label)
            display_abbr = meta_info["abbr"]
            display_full = meta_info["full"]
            color = meta_info["color"]
            
            ax.axvspan(start_time, end_time, facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Only draw text if segment is wide enough to avoid overlap
            if duration > (max_time * 0.015): 
                mid_time = (start_time + end_time) / 2
                ax.text(mid_time, 0.5, display_abbr, horizontalalignment='center', verticalalignment='center',
                        fontsize=10, fontweight='bold', color='black', transform=ax.get_xaxis_transform())
                
            if display_abbr not in legend_handles_dict:
                legend_handles_dict[display_abbr] = Line2D([0], [0], color='w', marker='s', markersize=10, 
                                                           markerfacecolor=color, label=f"{display_abbr}: {display_full}")
    
    # Sort legend
    sorted_legend_abbrs = sorted(legend_handles_dict.keys())
    legend_elements = [legend_handles_dict[abbr] for abbr in sorted_legend_abbrs]
    
    axes[0].legend(handles=legend_elements, loc='upper left', title="Formal Functions", 
                   bbox_to_anchor=(1.01, 1), fontsize='small')
    
    axes[1].set_xlabel('Time (s)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_caplin_contour(res_orig, res_cover, output_path: Path, meta_orig=None, meta_cover=None):
    from src.melody_analysis_v2.visualization import _get_caplin_meta
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.35)
    
    legend_handles_dict = {}
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        meta = meta_orig if i == 0 else meta_cover
        if meta:
            t_title, t_artist = meta
            ax.set_title(f"Melodic contour and detected segments - {name}\nSong: {t_title} | Performer: {t_artist}", fontweight='bold', fontsize=10)
        else:
            ax.set_title(f"Melodic contour and detected segments - {name}", fontweight='bold')
        
        # Pitch MIDI
        f0_midi = res.features.pitch_midi.copy()
        ax.plot(res.features.times, f0_midi, color='tab:blue', linewidth=1.5, label='Pitch (MIDI)')
        ax.set_ylabel('Pitch (MIDI)', color='tab:blue')
        
        # Energy
        ax2 = ax.twinx()
        energy = res.features.energy
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        ax2.plot(res.features.times, energy, color='tab:green', alpha=0.5, linewidth=1.0, label='Normalized energy')
        ax2.set_ylabel('Normalized energy', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_ylim(-0.05, 1.05)
        
        # Segments and Labels
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        for seg in res.segments:
            meta_info = _get_caplin_meta(seg.label)
            display_abbr = meta_info["abbr"]
            display_full = meta_info["full"]
            color = meta_info["color"]
            
            ax.axvspan(seg.segment.start_time, seg.segment.end_time, color=color, alpha=0.3)
            
            # Label on top (only if space)
            duration = seg.segment.end_time - seg.segment.start_time
            if duration > (max_time * 0.02):
                mid_time = (seg.segment.start_time + seg.segment.end_time) / 2
                ax.text(mid_time, 0.96, display_abbr, color='black', weight='bold', size=9,
                        horizontalalignment='center', transform=ax.get_xaxis_transform())
                        
            if display_abbr not in legend_handles_dict:
                from matplotlib.lines import Line2D
                legend_handles_dict[display_abbr] = Line2D([0], [0], color='w', marker='s', markersize=10, 
                                                            markerfacecolor=color, label=f"{display_abbr}: {display_full}")

        ax.set_xlim(0, max_time)
        
        # Legend
        if i == 0:
            sorted_legend_abbrs = sorted(legend_handles_dict.keys())
            legend_elements = [legend_handles_dict[abbr] for abbr in sorted_legend_abbrs]
            ax.legend(handles=legend_elements, loc='upper left', title="Formal Functions", 
                      bbox_to_anchor=(1.10, 1), fontsize='small')

    axes[1].set_xlabel('Time (s)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_contour_only_comparison(res_orig, res_cover, output_path: Path, meta_orig=None, meta_cover=None):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.35)
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        meta = meta_orig if i == 0 else meta_cover
        if meta:
            t_title, t_artist = meta
            ax.set_title(f"Melodic contour - {name}\nSong: {t_title} | Performer: {t_artist}", fontweight='bold', fontsize=10)
        else:
            ax.set_title(f"Melodic contour - {name}", fontweight='bold')
        
        # Pitch MIDI
        f0_midi = res.features.pitch_midi.copy()
        ax.plot(res.features.times, f0_midi, color='tab:blue', linewidth=1.5, label='Pitch (MIDI)')
        ax.set_ylabel('Pitch (MIDI)')
        
        # Twin axis for Hz scale representation
        ax2 = ax.twinx()
        f0_hz = np.nan_to_num(np.where(f0_midi > 0, 440.0 * np.power(2.0, (f0_midi - 69.0) / 12.0), 0))
        ax2.plot(res.features.times, f0_hz, color='tab:red', alpha=0.5, linewidth=1.0, label='f0 (Hz)')
        ax2.set_ylabel('f0 (Hz)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        ax.set_xlim(0, max_time)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel('Time (s)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_energy_only_comparison(res_orig, res_cover, output_path: Path, meta_orig=None, meta_cover=None):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.35)
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        meta = meta_orig if i == 0 else meta_cover
        if meta:
            t_title, t_artist = meta
            ax.set_title(f"Normalized energy - {name}\nSong: {t_title} | Performer: {t_artist}", fontweight='bold', fontsize=10)
        else:
            ax.set_title(f"Normalized energy - {name}", fontweight='bold')
        
        energy = res.features.energy
        ax.plot(res.features.times, energy, color='tab:green', linewidth=1.5)
        ax.set_ylabel('Normalized energy')
        
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        ax.set_xlim(0, max_time)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel('Time (s)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_melody_and_energy_comparison(res_orig, res_cover, output_path: Path, meta_orig=None, meta_cover=None):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.35)
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        meta = meta_orig if i == 0 else meta_cover
        if meta:
            t_title, t_artist = meta
            ax.set_title(f"Melodic contour and energy - {name}\nSong: {t_title} | Performer: {t_artist}", fontweight='bold', fontsize=10)
        else:
            ax.set_title(f"Melodic contour and energy - {name}", fontweight='bold')
        
        # Pitch MIDI
        f0_midi = res.features.pitch_midi.copy()
        ax.plot(res.features.times, f0_midi, color='tab:blue', linewidth=1.5, label='Pitch (MIDI)')
        ax.set_ylabel('Pitch (MIDI)', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        
        # Energy
        ax2 = ax.twinx()
        energy = res.features.energy
        ax2.plot(res.features.times, energy, color='tab:green', alpha=0.6, linewidth=1.2, label='Normalized energy')
        ax2.set_ylabel('Normalized energy', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        ax.set_xlim(0, max_time)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel('Time (s)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    script_dir = Path(__file__).parent.absolute()
    
    print("--- Melody Comparison (Original vs Cover) - Thesis Version ---")
    
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

    classifier = MelodyClassifierThesis()
    analyzer = MelodyAnalyzer(extraction_method=method, classifier=classifier)

    # Output directory for plots
    comp_dir = script_dir / "salidas_comparativa" / f"{path_orig.stem}_vs_{path_cover.stem}" / method
    
    print(f"\nAnalyzing Original: {path_orig.name}...")
    res_orig = analyzer.analyze_file(str(path_orig))
    save_plots(res_orig, path_orig, comp_dir / "original")
    
    print(f"Analyzing Cover: {path_cover.name}...")
    res_cover = analyzer.analyze_file(str(path_cover))
    save_plots(res_cover, path_cover, comp_dir / "cover")

    # Generate side-by-side comparison images
    print("\nGenerating side-by-side visual comparisons...")
    meta_orig = (path_orig.stem, "Original")
    meta_cover = (path_cover.stem, "Cover")
    
    plot_caplin_bands(res_orig, res_cover, comp_dir / "fig_qualitative_bands.png", meta_orig, meta_cover)
    plot_caplin_contour(res_orig, res_cover, comp_dir / "fig_qualitative_contour.png", meta_orig, meta_cover)
    plot_contour_only_comparison(res_orig, res_cover, comp_dir / "fig_qualitative_contour_only.png", meta_orig, meta_cover)
    plot_energy_only_comparison(res_orig, res_cover, comp_dir / "fig_qualitative_energy_only.png", meta_orig, meta_cover)
    plot_melody_and_energy_comparison(res_orig, res_cover, comp_dir / "fig_qualitative_contour_and_energy.png", meta_orig, meta_cover)

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
    print("\nLegend: A (Antecedent), C (Consequent), S (Rest/Silence)")

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
