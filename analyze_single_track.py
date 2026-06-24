#!/usr/bin/env python3
"""
MC-MSA v2.5: Analyze a Single Audio File
Extracts the melodic contour, detects structural segments, generates plots, and synthesizes the audio melody.
"""

import sys
import argparse
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Import the core MC-MSA pipeline tools
from src.melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodyClassifierPaper,
    MelodyClassifierPaperV2,
    MelodySegmenterBeta,
    synthesize_melody
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
    plot_descriptor_summary
)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a single audio file using the MC-MSA v2.5 pipeline.")
    parser.add_argument("audio_path", type=str, nargs="?", help="Path to the audio file (e.g., track.mp3 or track.wav)")
    parser.add_argument("--method", type=str, default="pyin",
                        choices=["pyin", "yin", "crepe", "rmvpe", "spice", "jdc", "fcn_f0", "melodia", "demucs_crepe", "bs_roformer_rmvpe", "basic_pitch"],
                        help="Melody extraction method (default: pyin)")
    parser.add_argument("--classifier", type=str, default="thesis", choices=["standard", "thesis", "thesis_v2"],
                        help="Structural classifier type (default: thesis)")
    parser.add_argument("--segmenter", type=str, default="thesis", choices=["thesis", "beta"],
                        help="Structural segmenter type (default: thesis)")
    parser.add_argument("--output_dir", type=str, default="salidas_single_track",
                        help="Directory to save the visualizations and synthesized melody")
    parser.add_argument("--save_txt", action="store_true", default=None,
                        help="Save textual results to a .txt report")
    parser.add_argument("--no_txt", action="store_false", dest="save_txt",
                        help="Do not save textual results to a .txt report")
    parser.add_argument("--save_plots", action="store_true", default=None,
                        help="Generate and save visualization images and synthesized audio")
    parser.add_argument("--no_plots", action="store_false", dest="save_plots",
                        help="Do not generate/save visualization images and synthesized audio")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve file path
    if not args.audio_path:
        audio_path_str = input("Enter path to the audio file: ").strip()
        if not audio_path_str:
            print("Error: No audio file provided.")
            sys.exit(1)
        audio_path = Path(audio_path_str)
    else:
        audio_path = Path(args.audio_path)
        
    if not audio_path.exists():
        print(f"Error: File not found at {audio_path}")
        sys.exit(1)
        
    print(f"\nAnalyzing: {audio_path.name}")
    print(f"Method: {args.method}")
    print(f"Classifier: {args.classifier}")
    
    # Configure Classifier
    classifier = None
    if args.classifier == "thesis":
        classifier = MelodyClassifierPaper()
        print("Using Thesis Classifier (A / C / X boundaries)")
    elif args.classifier == "thesis_v2":
        classifier = MelodyClassifierPaperV2()
        print("Using Thesis Classifier v2.0 (Corrected A / C / X boundaries)")
    else:
        print("Using Standard Caplin Classifier (Antecedent, Consequent, etc.)")
        
    # Configure Segmenter
    segmenter = None
    if args.segmenter == "beta":
        segmenter = MelodySegmenterBeta()
        print("Using Beta Segmenter (Hybrid SSM + local derivative novelty)")
    else:
        print("Using Thesis Segmenter (Pure SSM novelty)")
        
    # Initialize Melody Analyzer
    analyzer = MelodyAnalyzer(
        extraction_method=args.method,
        classifier=classifier,
        segmenter=segmenter
    )
    
    print("\nRunning melody extraction & segment analysis (this may take a moment)...")
    try:
        result = analyzer.analyze_file(str(audio_path))
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)
        
    # Print Segment Summary
    print("\n" + "="*50)
    print("           DETECTED FORMAL SEGMENTS")
    print("="*50)
    for idx, segment in enumerate(result.segments, 1):
        sim_score = segment.descriptor.get("ssm_similarity_with_previous", 0.0)
        print(f"[{idx:02d}] {segment.label:<20} | {segment.segment.start_time:7.2f}s -> {segment.segment.end_time:7.2f}s (SSM Sim: {sim_score:4.2f})")
    print("="*50 + "\n")
    
    # Setup Output Directory and Prompt Decisions
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    is_interactive = sys.stdin.isatty()
    
    if args.save_txt is None:
        if is_interactive:
            ans = input("Save results to a .txt report? (y/n) [y]: ").strip().lower()
            save_txt = ans != 'n'
        else:
            save_txt = True
    else:
        save_txt = args.save_txt

    if args.save_plots is None:
        if is_interactive:
            ans = input("Generate visualization images and synthesized audio? (y/n) [n]: ").strip().lower()
            save_plots = ans == 'y'
        else:
            save_plots = False
    else:
        save_plots = args.save_plots
        
    # Save Text Report
    if save_txt:
        txt_path = out_dir / f"{audio_path.stem}_results.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write("           DETECTED FORMAL SEGMENTS\n")
                f.write("="*50 + "\n")
                f.write(f"File: {audio_path.name}\n")
                f.write(f"Method: {args.method}\n")
                f.write(f"Classifier: {args.classifier}\n")
                f.write("="*50 + "\n")
                for idx, segment in enumerate(result.segments, 1):
                    sim_score = segment.descriptor.get("ssm_similarity_with_previous", 0.0)
                    f.write(f"[{idx:02d}] {segment.label:<20} | {segment.segment.start_time:7.2f}s -> {segment.segment.end_time:7.2f}s (SSM Sim: {sim_score:4.2f})\n")
                f.write("="*50 + "\n")
            print(f"Saved textual results to: {txt_path}")
        except Exception as e:
            print(f"Failed to save textual results: {e}")
            
    # Save Visualizations & Synthesis
    if save_plots:
        print("\nGenerating step-by-step visualizations...")
        
        # Load audio for spectrograms
        try:
            y, sr = sf.read(str(audio_path))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception as e:
            print(f"  - Warning: Could not load audio for spectrogram plots: {e}")
            y, sr = None, None

        # Step 1: f0 Contour
        try:
            fig = plot_melody_only(result, title="Step 1: f0 Contour")
            path = out_dir / f"{audio_path.stem}_step1_f0_contour.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [1/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [1/11] Failed: {e}")

        # Step 2: Normalized Energy
        try:
            fig = plot_energy_only(result, title="Step 2: Normalized Energy")
            path = out_dir / f"{audio_path.stem}_step2_energy.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [2/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [2/11] Failed: {e}")

        # Step 3: f0 Contour & Energy
        try:
            fig = plot_melody_and_energy(result, title="Step 3: f0 Contour & Energy")
            path = out_dir / f"{audio_path.stem}_step3_contour_energy.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [3/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [3/11] Failed: {e}")

        # Step 4: Mel-Spectrogram
        if y is not None:
            try:
                fig = plot_melspectrogram(y, sr, title="Step 4: Mel-Spectrogram")
                path = out_dir / f"{audio_path.stem}_step4_spectrogram.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  [4/11] Saved: {path.name}")
            except Exception as e:
                print(f"  [4/11] Failed: {e}")
        else:
            print("  [4/11] Skipped (Audio not loaded)")

        # Step 5: Self-Similarity Matrix (SSM)
        try:
            fig = plot_self_similarity(result, title="Step 5: Self-Similarity Matrix (SSM)")
            path = out_dir / f"{audio_path.stem}_step5_ssm.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [5/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [5/11] Failed: {e}")

        # Step 6: Novelty Curves & Boundary Detection
        try:
            fig = plot_boundary_detection(result, title="Step 6: Boundary Detection (Novelty Curves)")
            path = out_dir / f"{audio_path.stem}_step6_novelty_boundaries.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [6/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [6/11] Failed: {e}")

        # Step 7: Segment Extraction Bands (Visualized as blocks)
        try:
            fig = plot_segment_extraction(result, title="Step 7: Segment Extraction & Classification Bands")
            path = out_dir / f"{audio_path.stem}_step7_segmentation_bands.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [7/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [7/11] Failed: {e}")

        # Step 8: Segmented f0 Contour
        try:
            fig = plot_melody_contour(result, title="Step 8: Segmented f0 Contour")
            path = out_dir / f"{audio_path.stem}_step8_contour_segmented.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [8/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [8/11] Failed: {e}")

        # Step 9: Segmented Energy Contour
        try:
            fig = plot_energy_contour(result, title="Step 9: Segmented Normalized Energy")
            path = out_dir / f"{audio_path.stem}_step9_energy_segmented.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [9/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [9/11] Failed: {e}")

        # Step 10: Mel-Spectrogram with Segments
        if y is not None:
            try:
                fig = plot_spectrogram_with_segments(y, sr, result, title="Step 10: Mel-Spectrogram with Annotated Segments")
                path = out_dir / f"{audio_path.stem}_step10_spectrogram_segmented.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  [10/11] Saved: {path.name}")
            except Exception as e:
                print(f"  [10/11] Failed: {e}")
        else:
            print("  [10/11] Skipped (Audio not loaded)")

        # Step 11: Descriptor Summary (Bar plots per segment)
        try:
            metrics_to_show = ["pitch_slope", "pitch_range", "energy_mean", "energy_delta"]
            fig = plot_descriptor_summary(result, metrics=metrics_to_show)
            path = out_dir / f"{audio_path.stem}_step11_descriptor_summary.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [11/11] Saved: {path.name}")
        except Exception as e:
            print(f"  [11/11] Failed: {e}")

        # Synthesize Extracted Melody
        print("\nSynthesizing extracted melody...")
        try:
            synth_audio = synthesize_melody(
                result.features.times,
                result.features.pitch_midi,
                result.features.confidence,
                result.features.energy,
                sample_rate=22050
            )
            synth_path = out_dir / f"{audio_path.stem}_synthesized.wav"
            sf.write(str(synth_path), synth_audio, 22050)
            print(f"  - Synthesized melody audio: {synth_path}")
        except Exception as e:
            print(f"  - Could not synthesize melody: {e}")
            
    print(f"\nAnalysis completed successfully! Outputs saved in: {out_dir.absolute()}\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
