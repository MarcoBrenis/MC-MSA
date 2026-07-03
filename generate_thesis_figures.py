#!/usr/bin/env python3
"""
MC-MSA v2.5: Thesis SSM Figure Generator (Refined Pointer Edition)
Generates academic-quality figures representing the state of melody analysis before classification:
1. Raw SSM (ssm_1_raw.png)
2. SSM with Novelty Curve (ssm_2_novelty.png)
3. Boundaries (ssm_3_boundaries_unlabeled.png & ssm_3_boundaries_labeled.png)
4. Homogeneity (ssm_4_homogeneity_a_only.png, ssm_4_homogeneity_en.png, ssm_4_homogeneity_es.png)
5. Repetition (ssm_5_repetition_a_only.png, ssm_5_repetition_en.png, ssm_5_repetition_es.png)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

# Import the core MC-MSA pipeline tools
from src.melody_analysis_v2 import MelodyAnalyzer

def format_axes(ax, ssm_shape, times, step, use_time=False, is_x=True, is_es=False, offset=0, label=None):
    if use_time:
        def format_time(x, pos):
            idx = int(round(x)) + offset
            actual_idx = idx * step
            if 0 <= actual_idx < len(times):
                t = times[actual_idx]
                minutes = int(t // 60)
                seconds = t % 60
                if minutes > 0:
                    return f"{minutes}:{seconds:04.1f}"
                return f"{seconds:.1f}s"
            return ""
        
        if is_x:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
            if label is not None:
                ax.set_xlabel(label)
            else:
                ax.set_xlabel("Time" if not is_es else "Tiempo")
        else:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))
            if label is not None:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("Time (mm:ss)" if not is_es else "Tiempo (mm:ss)")
    else:
        if is_x:
            if label is not None:
                ax.set_xlabel(label)
            else:
                ax.set_xlabel("Frame Index" if not is_es else "Índice de Frame")
        else:
            if label is not None:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("Frame Index" if not is_es else "Índice de Frame")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis figures illustrating SSM concepts.")
    parser.add_argument("audio_path", type=str, nargs="?", help="Path to the audio file (e.g., 1.mp3)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["pyin", "spice"],
                        help="Melody extraction method to use (pyin or spice)")
    parser.add_argument("--output_dir", type=str, default="thesis_figures",
                        help="Directory to save the generated figures (default: thesis_figures)")
    return parser.parse_args()

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
        
    # Resolve melody extraction method
    if not args.method:
        print("\n=== Melody Extraction Method Selection ===")
        print("1. pyin")
        print("2. spice")
        while True:
            choice = input("Select melody extraction method (1 or 2, default: 1): ").strip()
            if not choice or choice == "1" or choice.lower() == "pyin":
                args.method = "pyin"
                break
            elif choice == "2" or choice.lower() == "spice":
                args.method = "spice"
                break
            else:
                print("Error: Invalid selection. Please enter 1, 2, 'pyin', or 'spice'.")
        
    # Create analyzer and run analysis
    print(f"\nAnalyzing '{audio_path.name}' using '{args.method}' to generate SSM...")
    analyzer = MelodyAnalyzer(extraction_method=args.method)
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
    
    # Extract novelty curves
    novelty = result.novelty
    base_novelty = result.base_novelty
    ssm_novelty = result.ssm_novelty
    
    # Calculate boundary indices in the SSM coordinates (pre-classification)
    boundaries = [0]
    for ann in result.segments:
        boundaries.append(ann.segment.end_index // step)
        
    # Keep boundaries within SSM limits
    boundaries = [min(b, ssm.shape[0]) for b in boundaries]
    # Remove duplicates
    boundaries = sorted(list(set(boundaries)))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set academic plotting style
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 13,
        'font.family': 'sans-serif'
    })
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Raw SSM (ssm_1_raw_frames.png / ssm_1_raw_time.png)
    # -------------------------------------------------------------------------
    print("\n1. Generating Figure 1: Raw Self-Similarity Matrix...")
    for use_time in [False, True]:
        suffix = "_time" if use_time else "_frames"
        fig, ax = plt.subplots(figsize=(7, 6))
        img = ax.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        ax.set_title("Raw Self-Similarity Matrix (SSM)")
        fig.colorbar(img, ax=ax, label="Similarity Score")
        fig.tight_layout()
        fig1_path = output_dir / f"ssm_1_raw{suffix}.png"
        fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig1_path.resolve()}")
    
    # -------------------------------------------------------------------------
    # FIGURE 2: SSM with Novelty Curve (ssm_2_novelty_frames.png / ssm_2_novelty_time.png)
    # -------------------------------------------------------------------------
    print("2. Generating Figure 2: SSM aligned with Novelty Curve...")
    for use_time in [False, True]:
        suffix = "_time" if use_time else "_frames"
        fig, (ax_nov, ax_ssm) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True, 
                                             gridspec_kw={'height_ratios': [1, 2.5]})
        
        # Top: Novelty Curves
        frames = np.arange(ssm.shape[0])
        if novelty is not None:
            ax_nov.plot(frames, novelty, color="#440154", linewidth=2.0, label="Combined Novelty")
        if base_novelty is not None:
            ax_nov.plot(frames, base_novelty, color="#31688e", linewidth=1.2, linestyle=":", label="Base Novelty (Derivative)")
        if ssm_novelty is not None:
            ax_nov.plot(frames, ssm_novelty, color="#35b779", linewidth=1.2, linestyle="--", label="SSM Novelty (Checkerboard)")
            
        # Draw vertical lines for boundaries and highlight peaks on the curve
        for b in boundaries[1:-1]:
            ax_nov.axvline(x=b, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            if novelty is not None and b < len(novelty):
                ax_nov.plot(b, novelty[b], "ro", markersize=5)
                
        ax_nov.set_ylabel("Novelty Score")
        ax_nov.set_title("A. Melodic Novelty Curve & Peak Detection")
        ax_nov.legend(loc="upper right", framealpha=0.9, fontsize=8)
        ax_nov.grid(True, linestyle=":", alpha=0.5)
        
        # Bottom: SSM aligned
        img = ax_ssm.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        for b in boundaries[1:-1]:
            ax_ssm.axvline(x=b, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_ssm.axhline(y=b, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            
        format_axes(ax_ssm, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax_ssm, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        ax_ssm.set_title("B. Aligned Self-Similarity Matrix (SSM)")
        
        fig.colorbar(img, ax=ax_ssm, label="Similarity Score", orientation="horizontal", pad=0.15, shrink=0.7)
        fig.tight_layout()
        fig2_path = output_dir / f"ssm_2_novelty{suffix}.png"
        fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig2_path.resolve()}")
    
    # -------------------------------------------------------------------------
    # FIGURE 3: SSM with Boundaries (ssm_3_boundaries_unlabeled & labeled)
    # -------------------------------------------------------------------------
    print("3. Generating Figure 3: SSM with Boundaries (Unlabeled & Labeled)...")
    for use_time in [False, True]:
        suffix = "_time" if use_time else "_frames"
        
        # 3.1 Unlabeled
        fig, ax = plt.subplots(figsize=(7, 6))
        img = ax.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(y=b, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        ax.set_title("SSM with Detected Structural Boundaries")
        fig.colorbar(img, ax=ax, label="Similarity Score")
        fig.tight_layout()
        fig3_unlabeled_path = output_dir / f"ssm_3_boundaries_unlabeled{suffix}.png"
        fig.savefig(fig3_unlabeled_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig3_unlabeled_path.resolve()}")
        
        # 3.2 Labeled
        fig, ax = plt.subplots(figsize=(7, 6))
        img = ax.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(y=b, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        ax.set_title("SSM with Detected Structural Boundaries")
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            mid = (start + end) / 2
            label = f"S{i+1}"
            if (end - start) > 12:
                ax.text(mid, mid, label, color="white", fontsize=9, ha="center", va="center", fontweight="bold",
                        bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.25', edgecolor='none'))
                
        fig.colorbar(img, ax=ax, label="Similarity Score")
        fig.tight_layout()
        fig3_labeled_path = output_dir / f"ssm_3_boundaries_labeled{suffix}.png"
        fig.savefig(fig3_labeled_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig3_labeled_path.resolve()}")
    
    # -------------------------------------------------------------------------
    # FIGURE 4: Homogeneity Principle (Homogeneity zoom-ins and A-only version)
    # -------------------------------------------------------------------------
    print("4. Generating Figure 4: Homogeneity Principle (A-only, EN, and ES)...")
    
    # Find a nice homogeneous block (diagonal block with highest internal coherence)
    best_diag_idx = -1
    best_diag_score = -1.0
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        if end - start < 15:
            continue
        block = ssm[start:end, start:end]
        coherence = np.mean(block)
        if coherence > best_diag_score:
            best_diag_score = coherence
            best_diag_idx = i
            
    # Default to the first reasonable block if none found
    if best_diag_idx == -1:
        for i in range(len(boundaries) - 1):
            if boundaries[i+1] - boundaries[i] > 5:
                best_diag_idx = i
                break
        if best_diag_idx == -1:
            best_diag_idx = 0
            
    h_start = boundaries[best_diag_idx]
    h_end = boundaries[best_diag_idx+1]
    h_len = h_end - h_start
    h_label = f"S{best_diag_idx+1}"
    
    # Dynamically place the homogeneity label box and arrow pointing to the block
    mid = h_start + h_len / 2
    if mid < ssm.shape[0] / 2:
        xy_target = (mid, h_start + h_len)
        xy_text = (mid, h_start + h_len + (ssm.shape[0] * 0.18))
    else:
        xy_target = (mid, h_start)
        xy_text = (mid, h_start - (ssm.shape[0] * 0.18))
        
    for use_time in [False, True]:
        suffix = "_time" if use_time else "_frames"
        
        # 4.1 A-only (Full SSM only)
        fig, ax = plt.subplots(figsize=(7, 6))
        img = ax.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title("A. Full Self-Similarity Matrix")
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axhline(y=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
        rect = patches.Rectangle(
            (h_start - 0.5, h_start - 0.5), h_len, h_len, linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        
        # Draw RED pointing arrow and RED-bordered text box
        ax.annotate(
            f"Homogeneous Block",
            xy=xy_target, xytext=xy_text,
            arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.08, width=2, headwidth=8, headlength=8),
            ha="center", va="center", fontsize=9.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", lw=2)
        )
        
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        
        fig.colorbar(img, ax=ax, label="Similarity Score")
        fig.tight_layout()
        fig4_a_path = output_dir / f"ssm_4_homogeneity_a_only{suffix}.png"
        fig.savefig(fig4_a_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig4_a_path.resolve()}")
        
        # 4.2 English Version (Full + Detail)
        fig = plt.figure(figsize=(10, 5.5))
        ax_full = plt.subplot2grid((1, 2), (0, 0))
        ax_zoom = plt.subplot2grid((1, 2), (0, 1))
        
        # Left: Full
        img1 = ax_full.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_full.set_title("A. Full Self-Similarity Matrix")
        for b in boundaries[1:-1]:
            ax_full.axvline(x=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_full.axhline(y=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
        rect = patches.Rectangle(
            (h_start - 0.5, h_start - 0.5), h_len, h_len, linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax_full.add_patch(rect)
        
        # Draw RED pointing arrow and RED-bordered text box
        ax_full.annotate(
            f"Homogeneous Block",
            xy=xy_target, xytext=xy_text,
            arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.08, width=2, headwidth=8, headlength=8),
            ha="center", va="center", fontsize=8.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2)
        )
        
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        
        # Right: Zoom
        zoom_block = ssm[h_start:h_end, h_start:h_end]
        img2 = ax_zoom.imshow(zoom_block, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_zoom.set_title(f"B. Detail: Diagonal Homogeneous Block ({h_label})")
        ax_zoom.plot([0, h_len-1], [0, h_len-1], color="red", linestyle=":", alpha=0.6, label="Main Diagonal")
        
        # Highlight zoom panel with red border as well
        rect_zoom = patches.Rectangle(
            (-0.5, -0.5), h_len, h_len, linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax_zoom.add_patch(rect_zoom)
        
        ax_zoom.text(h_len*0.05, h_len*0.85, "High intra-segment similarity\n(Internal thematic coherence)", 
                     color="white", fontsize=9, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        zoom_x_label = "Time (mm:ss)" if use_time else "Frame Index (Relative)"
        zoom_y_label = "Time (mm:ss)" if use_time else "Frame Index (Relative)"
        format_axes(ax_zoom, h_len, result.features.times, step, use_time=use_time, is_x=True, is_es=False, offset=h_start, label=zoom_x_label)
        format_axes(ax_zoom, h_len, result.features.times, step, use_time=use_time, is_x=False, is_es=False, offset=h_start, label=zoom_y_label)
        
        fig.colorbar(img1, ax=[ax_full, ax_zoom], label="Similarity Score", orientation="horizontal", pad=0.15, shrink=0.7)
        plt.suptitle("Homogeneity Principle: High Intra-Segment Self-Similarity", y=0.98, fontsize=13, fontweight="bold")
        fig4_en_path = output_dir / f"ssm_4_homogeneity_en{suffix}.png"
        fig.savefig(fig4_en_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig4_en_path.resolve()}")
        
        # 4.3 Spanish Version (Full + Detail)
        fig = plt.figure(figsize=(10, 5.5))
        ax_full = plt.subplot2grid((1, 2), (0, 0))
        ax_zoom = plt.subplot2grid((1, 2), (0, 1))
        
        # Left: Full (ES)
        img1 = ax_full.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_full.set_title("A. Matriz de Autosimilitud Completa")
        for b in boundaries[1:-1]:
            ax_full.axvline(x=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_full.axhline(y=b, color="white", linestyle="--", alpha=0.5, linewidth=0.8)
        rect = patches.Rectangle(
            (h_start - 0.5, h_start - 0.5), h_len, h_len, linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax_full.add_patch(rect)
        
        # Draw RED pointing arrow and RED-bordered text box
        ax_full.annotate(
            f"Bloque Homogéneo",
            xy=xy_target, xytext=xy_text,
            arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.08, width=2, headwidth=8, headlength=8),
            ha="center", va="center", fontsize=8.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2)
        )
        
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=True)
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=True)
        
        # Right: Zoom (ES)
        zoom_block = ssm[h_start:h_end, h_start:h_end]
        img2 = ax_zoom.imshow(zoom_block, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_zoom.set_title(f"B. Detalle: Bloque Homogéneo Diagonal ({h_label})")
        ax_zoom.plot([0, h_len-1], [0, h_len-1], color="red", linestyle=":", alpha=0.6, label="Diagonal Principal")
        
        rect_zoom = patches.Rectangle(
            (-0.5, -0.5), h_len, h_len, linewidth=2.5, edgecolor="red", facecolor="none"
        )
        ax_zoom.add_patch(rect_zoom)
        
        ax_zoom.text(h_len*0.05, h_len*0.85, "Alta similitud intra-segmento\n(Coherencia temática interna)", 
                     color="white", fontsize=9, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        zoom_x_label_es = "Tiempo (mm:ss)" if use_time else "Índice de Frame (Relativo)"
        zoom_y_label_es = "Tiempo (mm:ss)" if use_time else "Índice de Frame (Relativo)"
        format_axes(ax_zoom, h_len, result.features.times, step, use_time=use_time, is_x=True, is_es=True, offset=h_start, label=zoom_x_label_es)
        format_axes(ax_zoom, h_len, result.features.times, step, use_time=use_time, is_x=False, is_es=True, offset=h_start, label=zoom_y_label_es)
        
        fig.colorbar(img1, ax=[ax_full, ax_zoom], label="Similitud", orientation="horizontal", pad=0.15, shrink=0.7)
        plt.suptitle("Principio de Homogeneidad: Alta Auto-similitud Intra-segmento", y=0.98, fontsize=13, fontweight="bold")
        fig4_es_path = output_dir / f"ssm_4_homogeneity_es{suffix}.png"
        fig.savefig(fig4_es_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig4_es_path.resolve()}")
    
    # -------------------------------------------------------------------------
    # FIGURE 5: Repetition Principle (Repetition zoom-ins and A-only version)
    # -------------------------------------------------------------------------
    print("5. Generating Figure 5: Repetition Principle (A-only, EN, and ES)...")
    
    # Find the off-diagonal block with the highest average similarity (excluding diagonal)
    best_pair = None
    best_score = -1.0
    for i in range(len(boundaries) - 1):
        for j in range(len(boundaries) - 1):
            if abs(i - j) < 2:  # Skip diagonal/adjacent blocks
                continue
            start_i, end_i = boundaries[i], boundaries[i+1]
            start_j, end_j = boundaries[j], boundaries[j+1]
            
            # Avoid small blocks
            if (end_i - start_i) < 12 or (end_j - start_j) < 12:
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
    
    label_i = f"S{idx_i+1}"
    label_j = f"S{idx_j+1}"
    
    # Dynamically place the repetition match box and arrow pointing to the block
    mid_x = start_j + len_j/2
    mid_y = start_i + len_i/2
    if mid_y < ssm.shape[0]/2:
        xy_target_rep = (mid_x, start_i + len_i)
        xy_text_rep = (mid_x - (ssm.shape[0]*0.1), start_i + len_i + (ssm.shape[0]*0.18))
    else:
        xy_target_rep = (mid_x, start_i)
        xy_text_rep = (mid_x - (ssm.shape[0]*0.1), start_i - (ssm.shape[0]*0.18))
        
    for use_time in [False, True]:
        suffix = "_time" if use_time else "_frames"
        
        # 5.1 A-only (Full SSM with RED boxes)
        fig, ax = plt.subplots(figsize=(7, 6))
        img = ax.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_title("A. Structural Repetitions in SSM")
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
            ax.axhline(y=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
            
        rect_a1 = patches.Rectangle((start_i - 0.5, start_i - 0.5), len_i, len_i, linewidth=2, edgecolor="red", facecolor="none")
        rect_a2 = patches.Rectangle((start_j - 0.5, start_j - 0.5), len_j, len_j, linewidth=2, edgecolor="red", facecolor="none")
        rect_cross = patches.Rectangle((start_j - 0.5, start_i - 0.5), len_j, len_i, linewidth=2.5, edgecolor="red", facecolor="none", linestyle="--")
        
        ax.add_patch(rect_a1)
        ax.add_patch(rect_a2)
        ax.add_patch(rect_cross)
        
        ax.text(start_i + len_i/2, start_i + len_i/2, label_i, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
        ax.text(start_j + len_j/2, start_j + len_j/2, label_j, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
        
        # Draw RED pointing arrow and RED-bordered text box pointing to the matching block
        ax.annotate(
            f"Melodic Repetition\n({label_i} vs {label_j})",
            xy=xy_target_rep, xytext=xy_text_rep,
            arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.08, width=2, headwidth=8, headlength=8),
            ha="center", va="center", fontsize=9.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", lw=2)
        )
        
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        
        fig.colorbar(img, ax=ax, label="Similarity Score")
        fig.tight_layout()
        fig5_a_path = output_dir / f"ssm_5_repetition_a_only{suffix}.png"
        fig.savefig(fig5_a_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig5_a_path.resolve()}")
        
        # 5.2 English Version (Full + detail zoom-ins with RED boxes)
        fig = plt.figure(figsize=(11, 7.5))
        ax_full = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
        ax_a1 = plt.subplot2grid((2, 3), (0, 1))
        ax_a2 = plt.subplot2grid((2, 3), (1, 1))
        ax_cross = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)
        
        # Left
        img = ax_full.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_full.set_title("A. Structural Repetitions in SSM")
        for b in boundaries[1:-1]:
            ax_full.axvline(x=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
            ax_full.axhline(y=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
            
        rect_a1 = patches.Rectangle((start_i - 0.5, start_i - 0.5), len_i, len_i, linewidth=2, edgecolor="red", facecolor="none")
        rect_a2 = patches.Rectangle((start_j - 0.5, start_j - 0.5), len_j, len_j, linewidth=2, edgecolor="red", facecolor="none")
        rect_cross = patches.Rectangle((start_j - 0.5, start_i - 0.5), len_j, len_i, linewidth=2.5, edgecolor="red", facecolor="none", linestyle="--")
        ax_full.add_patch(rect_a1)
        ax_full.add_patch(rect_a2)
        ax_full.add_patch(rect_cross)
        
        ax_full.text(start_i + len_i/2, start_i + len_i/2, label_i, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
        ax_full.text(start_j + len_j/2, start_j + len_j/2, label_j, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
        
        # Draw RED pointing arrow and RED-bordered text box
        ax_full.annotate(
            f"Melodic Repetition\n({label_i} vs {label_j})",
            xy=xy_target_rep, xytext=xy_text_rep,
            arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.08, width=2, headwidth=8, headlength=8),
            ha="center", va="center", fontsize=8.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2)
        )
        
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=False)
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=False)
        
        # Zooms
        ax_a1.imshow(ssm[start_i:end_i, start_i:end_i], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_a1.set_title(f"B. Diagonal Block {label_i}")
        rect_z1 = patches.Rectangle((-0.5, -0.5), len_i, len_i, linewidth=2.5, edgecolor="red", facecolor="none")
        ax_a1.add_patch(rect_z1)
        
        zoom_x_label_a1 = "Time (mm:ss)" if use_time else "Relative Frame"
        zoom_y_label_a1 = "Time (mm:ss)" if use_time else "Relative Frame"
        format_axes(ax_a1, len_i, result.features.times, step, use_time=use_time, is_x=True, is_es=False, offset=start_i, label=zoom_x_label_a1)
        format_axes(ax_a1, len_i, result.features.times, step, use_time=use_time, is_x=False, is_es=False, offset=start_i, label=zoom_y_label_a1)
        
        ax_a2.imshow(ssm[start_j:end_j, start_j:end_j], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_a2.set_title(f"C. Diagonal Block {label_j}")
        rect_z2 = patches.Rectangle((-0.5, -0.5), len_j, len_j, linewidth=2.5, edgecolor="red", facecolor="none")
        ax_a2.add_patch(rect_z2)
        
        zoom_x_label_a2 = "Time (mm:ss)" if use_time else "Relative Frame"
        zoom_y_label_a2 = "Time (mm:ss)" if use_time else "Relative Frame"
        format_axes(ax_a2, len_j, result.features.times, step, use_time=use_time, is_x=True, is_es=False, offset=start_j, label=zoom_x_label_a2)
        format_axes(ax_a2, len_j, result.features.times, step, use_time=use_time, is_x=False, is_es=False, offset=start_j, label=zoom_y_label_a2)
        
        ax_cross.imshow(ssm[start_i:end_i, start_j:end_j], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_cross.set_title(f"D. Off-Diagonal Match ({label_i} vs {label_j})\n(Direct Structural Match)")
        rect_zc = patches.Rectangle((-0.5, -0.5), len_j, len_i, linewidth=2.5, edgecolor="red", facecolor="none")
        ax_cross.add_patch(rect_zc)
        
        ax_cross.text(len_j*0.05, len_i*0.8, "Shared similarity texture\nrepresents structural recurrence", 
                      color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        zoom_x_label_cross = f"Time of {label_j} (mm:ss)" if use_time else f"Frames of {label_j}"
        zoom_y_label_cross = f"Time of {label_i} (mm:ss)" if use_time else f"Frames of {label_i}"
        format_axes(ax_cross, len_j, result.features.times, step, use_time=use_time, is_x=True, is_es=False, offset=start_j, label=zoom_x_label_cross)
        format_axes(ax_cross, len_i, result.features.times, step, use_time=use_time, is_x=False, is_es=False, offset=start_i, label=zoom_y_label_cross)
        
        fig.colorbar(img, ax=[ax_full, ax_a1, ax_a2, ax_cross], label="Similarity Score", orientation="horizontal", pad=0.12, shrink=0.7)
        plt.suptitle("Repetition Principle: Matching Diagonal & Off-Diagonal SSM Patterns", y=0.98, fontsize=13, fontweight="bold")
        fig5_en_path = output_dir / f"ssm_5_repetition_en{suffix}.png"
        fig.savefig(fig5_en_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig5_en_path.resolve()}")
        
        # 5.3 Spanish Version (Full + detail zoom-ins with RED boxes)
        fig = plt.figure(figsize=(11, 7.5))
        ax_full = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
        ax_a1 = plt.subplot2grid((2, 3), (0, 1))
        ax_a2 = plt.subplot2grid((2, 3), (1, 1))
        ax_cross = plt.subplot2grid((2, 3), (0, 2), rowspan=2, colspan=1)
        
        # Left (ES)
        img = ax_full.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_full.set_title("A. Repeticiones Estructurales en la Matriz")
        for b in boundaries[1:-1]:
            ax_full.axvline(x=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
            ax_full.axhline(y=b, color="white", linestyle="--", alpha=0.4, linewidth=0.6)
            
        rect_a1 = patches.Rectangle((start_i - 0.5, start_i - 0.5), len_i, len_i, linewidth=2, edgecolor="red", facecolor="none")
        rect_a2 = patches.Rectangle((start_j - 0.5, start_j - 0.5), len_j, len_j, linewidth=2, edgecolor="red", facecolor="none")
        rect_cross = patches.Rectangle((start_j - 0.5, start_i - 0.5), len_j, len_i, linewidth=2.5, edgecolor="red", facecolor="none", linestyle="--")
        ax_full.add_patch(rect_a1)
        ax_full.add_patch(rect_a2)
        ax_full.add_patch(rect_cross)
        
        ax_full.text(start_i + len_i/2, start_i + len_i/2, label_i, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
        ax_full.text(start_j + len_j/2, start_j + len_j/2, label_j, color="white", ha="center", va="center", fontsize=8, fontweight="bold",
                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1', edgecolor='none'))
        
        # Draw RED pointing arrow and RED-bordered text box
        ax_full.annotate(
            f"Repetición Melódica\n({label_i} vs {label_j})",
            xy=xy_target_rep, xytext=xy_text_rep,
            arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.08, width=2, headwidth=8, headlength=8),
            ha="center", va="center", fontsize=8.5, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2)
        )
        
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=True, is_es=True)
        format_axes(ax_full, ssm.shape[0], result.features.times, step, use_time=use_time, is_x=False, is_es=True)
        
        # Zooms (ES)
        ax_a1.imshow(ssm[start_i:end_i, start_i:end_i], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_a1.set_title(f"B. Bloque Diagonal {label_i}")
        rect_z1 = patches.Rectangle((-0.5, -0.5), len_i, len_i, linewidth=2.5, edgecolor="red", facecolor="none")
        ax_a1.add_patch(rect_z1)
        
        zoom_x_label_a1_es = "Tiempo (mm:ss)" if use_time else "Frame Relativo"
        zoom_y_label_a1_es = "Tiempo (mm:ss)" if use_time else "Frame Relativo"
        format_axes(ax_a1, len_i, result.features.times, step, use_time=use_time, is_x=True, is_es=True, offset=start_i, label=zoom_x_label_a1_es)
        format_axes(ax_a1, len_i, result.features.times, step, use_time=use_time, is_x=False, is_es=True, offset=start_i, label=zoom_y_label_a1_es)
        
        ax_a2.imshow(ssm[start_j:end_j, start_j:end_j], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_a2.set_title(f"C. Bloque Diagonal {label_j}")
        rect_z2 = patches.Rectangle((-0.5, -0.5), len_j, len_j, linewidth=2.5, edgecolor="red", facecolor="none")
        ax_a2.add_patch(rect_z2)
        
        zoom_x_label_a2_es = "Tiempo (mm:ss)" if use_time else "Frame Relativo"
        zoom_y_label_a2_es = "Tiempo (mm:ss)" if use_time else "Frame Relativo"
        format_axes(ax_a2, len_j, result.features.times, step, use_time=use_time, is_x=True, is_es=True, offset=start_j, label=zoom_x_label_a2_es)
        format_axes(ax_a2, len_j, result.features.times, step, use_time=use_time, is_x=False, is_es=True, offset=start_j, label=zoom_y_label_a2_es)
        
        ax_cross.imshow(ssm[start_i:end_i, start_j:end_j], origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax_cross.set_title(f"D. Coincidencia Fuera de la Diagonal ({label_i} vs {label_j})\n(Coincidencia Estructural Directa)")
        rect_zc = patches.Rectangle((-0.5, -0.5), len_j, len_i, linewidth=2.5, edgecolor="red", facecolor="none")
        ax_cross.add_patch(rect_zc)
        
        ax_cross.text(len_j*0.05, len_i*0.8, "La textura de similitud compartida\nrepresenta recurrencia estructural", 
                      color="white", fontsize=8, bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        zoom_x_label_cross_es = f"Tiempo de {label_j} (mm:ss)" if use_time else f"Frames de {label_j}"
        zoom_y_label_cross_es = f"Tiempo de {label_i} (mm:ss)" if use_time else f"Frames de {label_i}"
        format_axes(ax_cross, len_j, result.features.times, step, use_time=use_time, is_x=True, is_es=True, offset=start_j, label=zoom_x_label_cross_es)
        format_axes(ax_cross, len_i, result.features.times, step, use_time=use_time, is_x=False, is_es=True, offset=start_i, label=zoom_y_label_cross_es)
        
        fig.colorbar(img, ax=[ax_full, ax_a1, ax_a2, ax_cross], label="Similitud", orientation="horizontal", pad=0.12, shrink=0.7)
        plt.suptitle("Principio de Repetición: Patrones de Autosimilitud Coincidentes", y=0.98, fontsize=13, fontweight="bold")
        fig5_es_path = output_dir / f"ssm_5_repetition_es{suffix}.png"
        fig.savefig(fig5_es_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved to: {fig5_es_path.resolve()}")
    
    print(f"\nAll thesis figure versions successfully generated in: {output_dir.absolute()}\n")

if __name__ == "__main__":
    main()
