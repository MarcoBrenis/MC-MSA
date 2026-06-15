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
    plot_melody_contour,
    plot_melody_and_energy,
    plot_spectrogram_with_segments,
    synthesize_melody
)
from src.melody_analysis_v2.visualization import (
    plot_self_similarity,
    plot_boundary_detection
)

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a single audio file using the MC-MSA v2.5 pipeline.")
    parser.add_argument("audio_path", type=str, nargs="?", help="Path to the audio file (e.g., track.mp3 or track.wav)")
    parser.add_argument("--method", type=str, default="pyin",
                        choices=["pyin", "yin", "crepe", "rmvpe", "spice", "jdc", "fcn_f0", "melodia", "demucs_crepe", "bs_roformer_rmvpe", "basic_pitch"],
                        help="Melody extraction method (default: pyin)")
    parser.add_argument("--classifier", type=str, default="standard", choices=["standard", "paper"],
                        help="Structural classifier type (default: standard)")
    parser.add_argument("--output_dir", type=str, default="salidas_single_track",
                        help="Directory to save the visualizations and synthesized melody")
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
    if args.classifier == "paper":
        classifier = MelodyClassifierPaper()
        print("Using Paper Classifier (A / C / X boundaries)")
    else:
        print("Using Standard Caplin Classifier (Antecedent, Consequent, etc.)")
        
    # Initialize Melody Analyzer
    analyzer = MelodyAnalyzer(extraction_method=args.method, classifier=classifier)
    
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
    
    # Setup Output Directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Melodic Contour Plot
    print("Generating visualizations...")
    contour_fig = plot_melody_and_energy(result)
    contour_path = out_dir / f"{audio_path.stem}_contour.png"
    contour_fig.savefig(contour_path, dpi=150, bbox_inches="tight")
    plt.close(contour_fig)
    print(f"  - Melody & Energy Contour plot: {contour_path}")
    
    # 2. Save Spectrogram with Segments
    try:
        y, sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        spec_fig = plot_spectrogram_with_segments(y, sr, result)
        spec_path = out_dir / f"{audio_path.stem}_spectrogram.png"
        spec_fig.savefig(spec_path, dpi=150, bbox_inches="tight")
        plt.close(spec_fig)
        print(f"  - Spectrogram with segments plot: {spec_path}")
    except Exception as e:
        print(f"  - Could not generate spectrogram plot: {e}")
        
    # 3. Save Self-Similarity Matrix (SSM)
    try:
        ssm_fig = plot_self_similarity(result)
        ssm_path = out_dir / f"{audio_path.stem}_ssm.png"
        ssm_fig.savefig(ssm_path, dpi=150, bbox_inches="tight")
        plt.close(ssm_fig)
        print(f"  - Self-similarity matrix plot: {ssm_path}")
    except Exception as e:
        print(f"  - Could not generate SSM plot: {e}")
        
    # 4. Synthesize Extracted Melody
    print("Synthesizing extracted melody...")
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
