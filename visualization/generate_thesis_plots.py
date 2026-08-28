#!/usr/bin/env python3
"""
MC-MSA v2.5: Thesis Figures Generator (Real-Audio Edition)
Uses a real audio file to generate conceptual figures illustrating:
1. Homogeneity Principle (diagonal cohesive blocks).
2. Repetition Principle (comparing matching diagonal and off-diagonal blocks).
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import the core MC-MSA pipeline tools
from src.melody_analysis_v2 import MelodyAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis figures using a real audio file.")
    parser.add_argument("audio_path", type=str, nargs="?", help="Path to the audio file (default: 1.mp3)")
    parser.add_argument("--method", type=str, default="pyin",
                        help="Melody extraction method to use (default: pyin)")
    parser.add_argument("--output_dir", type=str, default="thesis_figures",
                        help="Directory to save the generated figures (default: thesis_figures)")
    return parser.parse_args()

def analyze_audio(audio_path, method):
    """Runs the analysis pipeline on the audio file and extracts the SSM and boundaries."""
    print(f"\nAnalyzing '{audio_path.name}' using '{method}' to generate SSM...")
    analyzer = MelodyAnalyzer(extraction_method=method)
    try:
        result = analyzer.analyze_file(str(audio_path))
    except Exception as e:
        print(f"Error analyzing file: {e}")
        sys.exit(1)
        
    if result.self_similarity is None:
        print("Error: The analysis did not generate a Self-Similarity Matrix.")
        sys.exit(1)
        
    ssm = np.asarray(result.self_similarity)
    step = result.ssm_step
    
    # Calculate boundary indices in the SSM coordinates
    boundaries = [0]
    labels = []
    for ann in result.segments:
        boundaries.append(ann.segment.end_index // step)
        labels.append(ann.label)
        
    # Keep boundaries within SSM limits
    boundaries = [min(b, ssm.shape[0]) for b in boundaries]
    boundaries = sorted(list(set(boundaries)))
    
    return ssm, boundaries, labels

def plot_homogeneity_figure(ssm, boundaries, labels, output_path):
    """
    Generates a figure demonstrating the Homogeneity Principle using a real SSM.
    """
    # Find a nice homogeneous block (diagonal block with highest internal coherence)
    # We ignore very small blocks or silence blocks
    best_idx = -1
    best_score = -1.0
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        if end - start < 15:
            continue
        if i < len(labels) and any(s in labels[i].lower() for s in ["silence", "silencio", "x"]):
            continue
            
        block = ssm[start:end, start:end]
        coherence = np.mean(block)
        if coherence > best_score:
            best_score = coherence
            best_idx = i
            
    # Fallback to first block of length > 5 if none met criteria
    if best_idx == -1:
        for i in range(len(boundaries) - 1):
            if boundaries[i+1] - boundaries[i] > 5:
                best_idx = i
                break
        if best_idx == -1:
            best_idx = 0
            
    h_start = boundaries[best_idx]
    h_end = boundaries[best_idx+1]
    h_len = h_end - h_start
    h_label = labels[best_idx] if best_idx < len(labels) else f"Block {best_idx+1}"
    
    fig = plt.figure(figsize=(10, 5.5))
    ax_full = plt.subplot2grid((1, 2), (0, 0))
    ax_zoom = plt.subplot2grid((1, 2), (0, 1))
    
    # Left: Full SSM
    img1 = ax_full.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_full.set_xlabel("Frame Index")
    ax_full.set_ylabel("Frame Index")
    ax_full.set_title("A. Full Self-Similarity Matrix")
    
    # Draw all boundaries
    for b in boundaries[1:-1]:
        ax_full.axvline(x=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
        ax_full.axhline(y=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
        
    # Highlight the selected homogeneous block
    rect = patches.Rectangle(
        (h_start, h_start), h_len, h_len, linewidth=2.5, edgecolor="lime", facecolor="none"
    )
    ax_full.add_patch(rect)
    ax_full.text(h_start + h_len/2, h_start + h_len/2, f"Homogeneous\nBlock\n({h_label})", 
                 color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2', edgecolor='none'))
    
    # Right: Zoom-in on the selected homogeneous block
    zoom_block = ssm[h_start:h_end, h_start:h_end]
    img2 = ax_zoom.imshow(zoom_block, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_zoom.set_xlabel("Frame Index (Relative)")
    ax_zoom.set_ylabel("Frame Index (Relative)")
    ax_zoom.set_title(f"B. Detail: Diagonal Homogeneous Block ({h_label})")
    
    ax_zoom.plot([0, h_len-1], [0, h_len-1], color="red", linestyle=":", alpha=0.5, label="Main Diagonal")
    
    # Add annotation text explaining the principle
    ax_zoom.text(h_len*0.05, h_len*0.85, "High intra-segment similarity\n(Internal thematic coherence)", 
                 color="white", fontsize=9, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    fig.colorbar(img1, ax=[ax_full, ax_zoom], label="Similarity Score", orientation="horizontal", pad=0.15, shrink=0.7)
    
    plt.suptitle("Homogeneity Principle: High Intra-Segment Self-Similarity", y=0.98, fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_repetition_figure(ssm, boundaries, labels, output_path):
    """
    Generates a figure demonstrating the Repetition Principle using a real SSM.
    """
    # Find the off-diagonal block with the highest average similarity (excluding diagonal)
    best_pair = None
    best_score = -1.0
    
    for i in range(len(boundaries) - 1):
        for j in range(len(boundaries) - 1):
            if abs(i - j) < 2:  # Skip diagonal/adjacent blocks
                continue
            start_i, end_i = boundaries[i], boundaries[i+1]
            start_j, end_j = boundaries[j], boundaries[j+1]
            
            # Avoid small blocks or silence blocks
            if (end_i - start_i) < 12 or (end_j - start_j) < 12:
                continue
            if i < len(labels) and any(s in labels[i].lower() for s in ["silence", "silencio", "x"]):
                continue
            if j < len(labels) and any(s in labels[j].lower() for s in ["silence", "silencio", "x"]):
                continue
                
            block = ssm[start_i:end_i, start_j:end_j]
            score = np.mean(block)
            if score > best_score:
                best_score = score
                best_pair = (i, j)
                
    # Fallback to largest blocks if no off-diagonal matched criteria
    if best_pair is None:
        large_indices = np.argsort([boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)])[::-1]
        if len(large_indices) >= 2:
            best_pair = (large_indices[0], large_indices[1])
        else:
            best_pair = (0, min(1, len(boundaries)-2))
            
    idx_i, idx_j = best_pair
    start_i, end_i = boundaries[idx_i], boundaries[idx_i+1]
    start_j, end_j = boundaries[idx_j], boundaries[idx_j+1]
    
    len_i = end_i - start_i
    len_j = end_j - start_j
    
    label_i = labels[idx_i] if idx_i < len(labels) else f"Block {idx_i+1}"
    label_j = labels[idx_j] if idx_j < len(labels) else f"Block {idx_j+1}"
    
    fig = plt.figure(figsize=(11, 7.5))
    ax_full = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
    ax_a1 = plt.subplot2grid((2, 3), (0, 1))
    ax_a2 = plt.subplot2grid((2, 3), (1, 1))
    ax_cross = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)
    
    # Left: Full SSM
    img = ax_full.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_full.set_xlabel("Frame Index")
    ax_full.set_ylabel("Frame Index")
    ax_full.set_title("A. Structural Repetitions in SSM")
    
    # Draw all boundaries
    for b in boundaries[1:-1]:
        ax_full.axvline(x=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
        ax_full.axhline(y=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
        
    # Highlight matching diagonal blocks and their off-diagonal match
    rect_a1 = patches.Rectangle((start_i, start_i), len_i, len_i, linewidth=2, edgecolor="orange", facecolor="none")
    rect_a2 = patches.Rectangle((start_j, start_j), len_j, len_j, linewidth=2, edgecolor="cyan", facecolor="none")
    rect_cross = patches.Rectangle((start_j, start_i), len_j, len_i, linewidth=2.5, edgecolor="magenta", facecolor="none", linestyle="--")
    
    ax_full.add_patch(rect_a1)
    ax_full.add_patch(rect_a2)
    ax_full.add_patch(rect_cross)
    
    ax_full.text(start_i + len_i/2, start_i + len_i/2, label_i, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
    ax_full.text(start_j + len_j/2, start_j + len_j/2, label_j, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
    ax_full.text(start_j + len_j/2, start_i + len_i/2, "Match", color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.1', edgecolor='none'))
    
    # Right 1: Zoom-in on Diagonal Block 1
    ax_a1.imshow(ssm[start_i:end_i, start_i:end_i], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_a1.set_title(f"B. Diagonal Block {label_i}")
    ax_a1.set_xlabel("Relative Frame")
    ax_a1.set_ylabel("Relative Frame")
    
    # Right 2: Zoom-in on Diagonal Block 2
    ax_a2.imshow(ssm[start_j:end_j, start_j:end_j], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_a2.set_title(f"C. Diagonal Block {label_j}")
    ax_a2.set_xlabel("Relative Frame")
    ax_a2.set_ylabel("Relative Frame")
    
    # Far Right: Zoom-in on the Off-Diagonal Block (Intersection of both)
    ax_cross.imshow(ssm[start_i:end_i, start_j:end_j], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_cross.set_title(f"D. Off-Diagonal Match ({label_i} vs {label_j})\n(Direct Structural Match)")
    ax_cross.set_xlabel(f"Frames of {label_j}")
    ax_cross.set_ylabel(f"Frames of {label_i}")
    
    ax_cross.text(len_j*0.05, len_i*0.8, "Shared similarity texture\nrepresents structural recurrence", 
                  color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    
    fig.colorbar(img, ax=[ax_full, ax_a1, ax_a2, ax_cross], label="Similarity Score", orientation="horizontal", pad=0.12, shrink=0.7)
    
    plt.suptitle("Repetition Principle: Matching Diagonal & Off-Diagonal SSM Patterns", y=0.98, fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    
    # Resolve file path
    if not args.audio_path:
        audio_path_str = input("Enter path to the audio file (default: 1.mp3): ").strip()
        if not audio_path_str:
            audio_path_str = "1.mp3"
        audio_path = Path(audio_path_str)
    else:
        audio_path = Path(args.audio_path)
        
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        sys.exit(1)
        
    # Analyze real audio file
    ssm, boundaries, labels = analyze_audio(audio_path, args.method)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate Homogeneity illustration
    fig_hom_path = output_dir / "fig_homogeneity.png"
    print(f"Generating Homogeneity figure: {fig_hom_path}...")
    plot_homogeneity_figure(ssm, boundaries, labels, fig_hom_path)
    
    # Generate Repetition illustration
    fig_rep_path = output_dir / "fig_repetition.png"
    print(f"Generating Repetition figure: {fig_rep_path}...")
    plot_repetition_figure(ssm, boundaries, labels, fig_rep_path)
    
    print(f"\nSuccessfully generated thesis figures using '{audio_path.name}' in: {output_dir.absolute()}")
    print("- fig_homogeneity.png")
    print("- fig_repetition.png\n")

if __name__ == "__main__":
    main()
