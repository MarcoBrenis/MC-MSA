import os
import re
import json
import argparse
import gc
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from src.melody_analysis_v2 import MelodyAnalyzer, MelodyClassifierThesisBeta, MelodySegmenterBeta, MelodyFeatures, MelodySegmentAnnotation, DiagramExporter
from src.melody_analysis_v2.classifier_thesis import calculate_lcs
from src.melody_analysis_v2.segmenter import MelodySegment
from src.melody_analysis_v2.pipeline import MelodyAnalysisResult
from src.melody_analysis_v2.visualization import (
    plot_boundary_detection, 
    plot_self_similarity, 
    plot_spectrogram_with_segments
)

METHOD_CLASSIFICATION = {
    'pyin': 'Fundamental Frequency (F0 Extractor)',
    'yin': 'Fundamental Frequency (F0 Extractor)',
    'crepe': 'Fundamental Frequency (F0 Extractor)',
    'rmvpe': 'Fundamental Frequency (F0 Extractor)',
    'spice': 'Fundamental Frequency (F0 Extractor)',
    'jdc': 'Fundamental Frequency (F0 Extractor)',
    'fcn_f0': 'Fundamental Frequency (F0 Extractor)',
    'melodia': 'Melody Extractor (Melody Extractor - Salamon)',
    'demucs_crepe': 'Melody Extractor (Melody Extractor - Demucs+CREPE)',
    'bs_roformer_rmvpe': 'Melody Extractor (Melody Extractor - Roformer+RMVPE)',
    'bs_roformer': 'Melody Extractor (Melody Extractor - Roformer+RMVPE)',
    'demucs': 'Melody Extractor (Melody Extractor - Demucs+CREPE)',
    'ensemble': 'Hybrid (Ensemble F0/Melody)',
    'all': 'All methods'
}

def get_audio_files(directory_path: Path, match_mode: str = "id"):
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            if match_mode == "stem":
                # Normalize the file stem
                name = f.stem.lower()
                # Remove common suffixes
                name = re.sub(r'[-_](cover|originales|original|orig|ref|covers|version|var|originales_clei)', '', name)
                # Remove numeric prefixes if present
                name = re.sub(r'^\d+\s*[-_]?\s*', '', name)
                # Keep only alphanumeric characters
                name = re.sub(r'[^a-z0-9]', '', name)
                key = name.strip()
                if key:
                    result[key] = f
            elif match_mode == "fuzzy":
                # Save by full name (guaranteed unique)
                result[f.name] = f
            else: # "id"
                match = re.search(r'^(\d+)', f.name)
                if match:
                    file_id = int(match.group(1))
                    result[file_id] = f
    return result

def pair_files_fuzzy(orig_files: dict[str, Path], cover_files: dict[str, Path]) -> tuple[dict[int, Path], dict[int, Path]]:
    """
    Pairs originals and covers using prefix extraction and fuzzy name matching.
    Returns two dictionaries mapping a unique numeric index (1 to N) to paired Paths.
    """
    paired_orig = {}
    paired_cover = {}
    
    # 1. Group by numeric prefix (or group 0 by default)
    def get_prefix_and_text(filename: str):
        match = re.match(r'^(\d+)\s*[-_]?\s*(.*)', filename)
        if match:
            return int(match.group(1)), match.group(2)
        return 0, filename
        
    orig_by_prefix = {}
    cover_by_prefix = {}
    
    for name, path in orig_files.items():
        prefix, text = get_prefix_and_text(name)
        orig_by_prefix.setdefault(prefix, []).append((text, path))
            
    for name, path in cover_files.items():
        prefix, text = get_prefix_and_text(name)
        cover_by_prefix.setdefault(prefix, []).append((text, path))
            
    # 2. Match for each prefix
    pair_id = 1
    
    def word_overlap(str1: str, str2: str) -> float:
        w1 = set(re.findall(r'[a-zA-Z0-9]+', str1.lower()))
        w2 = set(re.findall(r'[a-zA-Z0-9]+', str2.lower()))
        # Remove common stopwords
        stopwords = {'cover', 'covers', 'original', 'originales', 'orig', 'ref', 'version', 'mp3', 'wav'}
        w1 = w1 - stopwords
        w2 = w2 - stopwords
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))
        
    for prefix in sorted(orig_by_prefix.keys()):
        if prefix in cover_by_prefix:
            orig_list = orig_by_prefix[prefix]
            cover_list = cover_by_prefix[prefix]
            
            used_origs = set()
            for cov_text, cov_path in cover_list:
                best_overlap = -1.0
                best_orig_path = None
                
                for orig_text, orig_path in orig_list:
                    if orig_path in used_origs:
                        continue
                    overlap = word_overlap(cov_text, orig_text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_orig_path = orig_path
                        
                if best_orig_path is not None and best_overlap > 0.0:
                    paired_orig[pair_id] = best_orig_path
                    paired_cover[pair_id] = cov_path
                    used_origs.add(best_orig_path)
                    pair_id += 1
                    
    return paired_orig, paired_cover

def calculate_levenshtein_similarity(seq1, seq2) -> float:
    """Calculates normalized Levenshtein similarity between two label sequences."""
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # Deletion
                    dp[i][j-1] + 1,    # Insertion
                    dp[i-1][j-1] + 1   # Substitution
                )
    distance = dp[n][m]
    max_len = max(n, m)
    return 1.0 - (distance / max_len)

def calculate_pitch_histogram_similarity(pitch1, pitch2) -> float:
    """Calculates cosine similarity of pitch class histograms (chroma)."""
    p1_valid = pitch1[~np.isnan(pitch1) & (pitch1 > 0)]
    p2_valid = pitch2[~np.isnan(pitch2) & (pitch2 > 0)]
    
    hist1 = np.zeros(12)
    hist2 = np.zeros(12)
    
    if len(p1_valid) > 0:
        classes1 = np.round(p1_valid).astype(int) % 12
        for pc in classes1:
            hist1[pc] += 1
        norm1 = np.linalg.norm(hist1)
        if norm1 > 0:
            hist1 = hist1 / norm1
            
    if len(p2_valid) > 0:
        classes2 = np.round(p2_valid).astype(int) % 12
        for pc in classes2:
            hist2[pc] += 1
        norm2 = np.linalg.norm(hist2)
        if norm2 > 0:
            hist2 = hist2 / norm2
            
    if np.linalg.norm(hist1) == 0 or np.linalg.norm(hist2) == 0:
        return 0.0
        
    return float(np.dot(hist1, hist2))

def evaluate_binary_classification(pairwise_results, metric_name, lower_is_better=False):
    """Evaluates binary classification for a given metric over a range of thresholds."""
    valid_results = [r for r in pairwise_results if r[0] is not None and r[0] >= 0]
    if not valid_results:
        return 0.0, None, []
        
    values = [r[0] for r in valid_results]
    min_val, max_val = min(values), max(values)
    
    if lower_is_better:
        thresholds = np.linspace(min_val, max_val, 21)
    else:
        thresholds = np.linspace(0.0, 1.0, 21)
        
    best_f1 = -1.0
    best_thresh = 0.0
    best_metrics = {}
    curves = []
    
    for t in thresholds:
        tp, fp, fn, tn = 0, 0, 0, 0
        for val, is_correct in valid_results:
            if lower_is_better:
                pred_positive = (val <= t)
            else:
                pred_positive = (val >= t)
                
            if pred_positive:
                if is_correct:
                    tp += 1
                else:
                    fp += 1
            else:
                if is_correct:
                    fn += 1
                else:
                    tn += 1
                    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        curves.append({
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            }
            
    return best_thresh, best_metrics, curves


def load_or_analyze(analyzer, file_path, method, cache_dir):
    cache_path = cache_dir / method / f"{file_path.stem}.json"
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            features = MelodyFeatures(
                times=np.array(data["times"]),
                pitch_midi=np.array(data["pitch_midi"]),
                confidence=np.array(data["confidence"]),
                energy=np.array(data["energy"])
            )
            # Re-run segmentation and classification dynamically using the cached features
            result = analyzer.analyze_features(features)
            # Save updated classifications back to cache JSON
            with open(cache_path, 'w') as f:
                json.dump(result.to_dict(), f)
            return result
        except Exception:
            pass
    
    result = analyzer.analyze_file(str(file_path))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(result.to_dict(), f)
    return result

def load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=""):
    import subprocess
    import sys
    cache_dir_method = cache_dir / method
    tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
    
    if tiny_path.exists():
        try:
            with open(tiny_path, 'r') as f:
                data = json.load(f)
            return {
                'seq': data['seq'],
                'pitch_midi': np.array(data['pitch_midi'])
            }
        except Exception:
            pass
            
    # If tiny does not exist, use subprocess pipeline to avoid memory leaks/OOM
    script_path = Path(__file__).parent / "src" / "melody_analysis_v2" / "analyze_single.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--file_path", str(file_path),
        "--method", method,
        "--cache_dir", str(cache_dir),
        "--label_prefix", label_prefix
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
    
    # Once completed, the normal cache .json is already saved on disk.
    res = load_or_analyze(analyzer, file_path, method, cache_dir)
    seq = [s.label for s in res.segments]
    pitch_midi = res.features.pitch_midi.copy() if res.features.pitch_midi is not None else None
    
    # Write the tiny json
    tiny_data = {
        'seq': seq,
        'pitch_midi': pitch_midi.tolist() if pitch_midi is not None else []
    }
    try:
        tiny_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tiny_path, 'w') as f:
            json.dump(tiny_data, f)
    except Exception:
        pass
        
    del res
    gc.collect()
        
    return {
        'seq': seq,
        'pitch_midi': pitch_midi
    }


def plot_caplin_bands(res_orig, res_cover, output_path: Path):
    from matplotlib.lines import Line2D
    from src.melody_analysis_v2.visualization import _get_caplin_meta
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    plt.subplots_adjust(hspace=0.6)
    
    legend_handles_dict = {}
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        ax.set_title(f"Melodic Segmentation (Bands) - {name}", fontweight='bold')
        ax.set_yticks([])
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        ax.set_xlim(0, max_time)
        
        for seg in res.segments:
            start_time = seg.segment.start_time
            end_time = seg.segment.end_time
            label = seg.label
            duration = end_time - start_time
            
            meta = _get_caplin_meta(label)
            display_abbr = meta["abbr"]
            display_full = meta["full"]
            color = meta["color"]
            
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
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    # Save a PNG version too
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_caplin_contour(res_orig, res_cover, output_path: Path):
    from src.melody_analysis_v2.visualization import _get_caplin_meta
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    legend_handles_dict = {}
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
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
            meta = _get_caplin_meta(seg.label)
            display_abbr = meta["abbr"]
            display_full = meta["full"]
            color = meta["color"]
            
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
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    # Save a PNG version too
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_contour_only_comparison(res_orig, res_cover, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
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
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    # Save a PNG version too
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_energy_only_comparison(res_orig, res_cover, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
        ax.set_title(f"Normalized energy - {name}", fontweight='bold')
        
        energy = res.features.energy
        ax.plot(res.features.times, energy, color='tab:green', linewidth=1.5)
        ax.set_ylabel('Normalized energy')
        
        max_time = res.features.times[-1] if len(res.features.times) > 0 else 1.0
        ax.set_xlim(0, max_time)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel('Time (s)')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_melody_and_energy_comparison(res_orig, res_cover, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    for i, (name, res) in enumerate([('Original', res_orig), ('Cover', res_cover)]):
        ax = axes[i]
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
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    available_methods = [
        'all', 'pyin', 'yin', 'melodia', 'crepe', 'ensemble', 
        'demucs_crepe', 'bs_roformer_rmvpe', 'rmvpe',
        'bs_roformer', 'demucs', 'jdc', 'spice', 'fcn_f0'
    ]
    parser = argparse.ArgumentParser(description="MC-MSA evaluation for melody extraction with cache support.")
    parser.add_argument("--method", type=str, default=None, 
                        choices=available_methods,
                        help="Extraction method to use")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Base directory of the dataset")
    parser.add_argument("--orig_subdir", type=str, default="originales",
                        help="Subdirectory of original songs")
    parser.add_argument("--cover_subdir", type=str, default="covers",
                        help="Subdirectory of cover songs")
    parser.add_argument("--output_dir", type=str, default="resultados_mc_msa",
                        help="Directory for graphical outputs and reports (if relative, resolves inside dataset folder)")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory for JSON analysis cache")
    parser.add_argument("--match_mode", type=str, default=None,
                        choices=["id", "stem"],
                        help="Match method: 'id' (numeric ID prefix) or 'stem' (normalized name/stem)")
    parser.add_argument("--dtw_all_pairs", action="store_true",
                        help="Compute DTW for all pairs (slow)")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Delete existing cache for the selected method before starting")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()

    # Search for available datasets dynamically
    def find_available_datasets(directory: Path):
        datasets = []
        if not directory.exists():
            return datasets
        for item in directory.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                has_orig_cov = (item / "originales").exists() and (item / "covers").exists()
                if item.name.startswith("dataset_") or has_orig_cov:
                    datasets.append(item.name)
        return sorted(list(set(datasets)))

    if args.dataset_dir is None:
        datasets = find_available_datasets(base_dir)
        if not datasets:
            print("\nNo dataset folders automatically detected in the base directory.")
            manual = input("Please enter the path or name of the dataset to use: ").strip()
            args.dataset_dir = manual
        else:
            print("\n=== Dataset Selection ===")
            for i, d in enumerate(datasets, 1):
                print(f"{i}. {d}")
            print(f"{len(datasets) + 1}. [Enter another manual path...]")
            
            while True:
                try:
                    choice = input(f"\nSelect a dataset (1-{len(datasets) + 1}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(datasets):
                        args.dataset_dir = datasets[idx]
                        break
                    elif idx == len(datasets):
                        manual = input("Enter the path or name of the dataset: ").strip()
                        if manual:
                            args.dataset_dir = manual
                            break
                    else:
                        print(f"Error: Please select a number between 1 and {len(datasets) + 1}.")
                except ValueError:
                    if choice in datasets:
                        args.dataset_dir = choice
                        break
                    print("Error: Invalid input. Enter option number or exact name.")

    if args.method is None:
        print("\n=== Extraction Method Selection ===")
        for i, m in enumerate(available_methods, 1):
            classification = METHOD_CLASSIFICATION.get(m, "")
            label = f" [{classification}]" if classification else ""
            print(f"{i}. {m}{label}")
        
        while True:
            try:
                choice = input(f"\nSelect a method (1-{len(available_methods)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(available_methods):
                    args.method = available_methods[idx]
                    break
                else:
                    print(f"Error: Please select a number between 1 and {len(available_methods)}.")
                    gc.collect()
            except ValueError:
                if choice.strip().lower() in available_methods:
                    args.method = choice.strip().lower()
                    break
                print("Error: Invalid input. Enter method number or name.")
                gc.collect()

    if args.match_mode is None:
        print("\n=== Match Method Selection ===")
        print("1. By Numeric ID (e.g. '01 - Pedro Infante.wav' with '01 - Cover.mp3')")
        print("2. By Exact Name / Stem (e.g. 'Te_Vi_Venir_Original.wav' with 'Te Vi Venir (Covers).mp3')")
        print("3. Smart / Fuzzy Match (For complex names or classical music, e.g. '02 - Symphony No. 40...')")
        
        while True:
            choice = input("\nSelect match method (1-3) [Default: 1]: ").strip()
            if not choice or choice == "1":
                args.match_mode = "id"
                break
            elif choice == "2":
                args.match_mode = "stem"
                break
            elif choice == "3":
                args.match_mode = "fuzzy"
                break
            else:
                print("Error: Please select 1, 2 or 3.")
                
    args.match_by_stem = (args.match_mode == "stem")

    # Resolve paths flexibly
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = base_dir / dataset_dir
        
    orig_dir = dataset_dir / args.orig_subdir
    cover_dir = dataset_dir / args.cover_subdir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = dataset_dir / output_dir
        
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir
        
    # Optional cache clearing (CLI or interactive)
    method_cache_dir = cache_dir / args.method
    if args.clear_cache:
        if method_cache_dir.exists():
            print(f"\n[Cache] Deleting existing cache files for method '{args.method}' in {method_cache_dir}...")
            import shutil
            shutil.rmtree(method_cache_dir)
            print("[Cache] Deletion completed.")
    else:
        if method_cache_dir.exists() and method_cache_dir.is_dir() and any(method_cache_dir.iterdir()):
            ans = input(f"\nDo you want to delete the existing cache for the method '{args.method}' before starting? (y/n): ").strip().lower()
            if ans in ['s', 'si', 'y', 'yes']:
                print(f"[Cache] Deleting cache files at {method_cache_dir}...")
                import shutil
                shutil.rmtree(method_cache_dir)
                print("[Cache] Deletion completed.")
    
    if not orig_dir.exists() or not cover_dir.exists():
        print(f"Source directories '{orig_dir}' and/or '{cover_dir}' do not exist.")
        return
        
    orig_files = get_audio_files(orig_dir, match_mode=args.match_mode)
    cover_files = get_audio_files(cover_dir, match_mode=args.match_mode)
    
    if args.match_mode == "fuzzy":
        orig_files, cover_files = pair_files_fuzzy(orig_files, cover_files)
        common_ids = sorted(list(orig_files.keys()))
    else:
        common_ids = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
    
    if args.method == 'all':
        methods = [m for m in available_methods if m != 'all']
    else:
        methods = [args.method]

    print(f"Total pairs (Original-Cover) found: {len(common_ids)}")
    print("Methods to evaluate:")
    for m in methods:
        classification = METHOD_CLASSIFICATION.get(m, "Unknown")
        print(f"  - {m}: {classification}")

    segmenter = MelodySegmenterBeta()
    classifier = MelodyClassifierThesisBeta()
    summary_path = output_dir / "mc_msa_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Check if summary_path exists but has old format, delete it to avoid column mismatch
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                first_line = f.readline().strip()
            if "mr" not in first_line or "mdr" not in first_line:
                summary_path.unlink()
        except Exception:
            pass

    if not summary_path.exists():
        with open(summary_path, 'w') as f:
            f.write("method,pairs,avg_lcs,mr,mrr,mdr,map,top5_prec,avg_dtw\n")

    for method in methods:
        classification = METHOD_CLASSIFICATION.get(method, "Unknown")
        print(f"\n[{method}] Processing... ({classification})")
        # Initialize Melody Analyzer for this method
        analyzer = MelodyAnalyzer(
            extraction_method=method,
            classifier=classifier,
            segmenter=segmenter
        )
        
        out_method_dir = output_dir / method
        out_method_dir.mkdir(parents=True, exist_ok=True)

        res_originals = {}
        total_p = len(common_ids)
        for i, uid in enumerate(common_ids, 1):
            try:
                file_path = orig_files[uid]
                prefix = f"  [{i}/{total_p}] ({i/total_p:.1%}) [Original] [{method}]"
                cache_dir_method = cache_dir / method
                tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
                if tiny_path.exists():
                    print(f"{prefix} [Cache] {file_path.name}")
                else:
                    print(f"{prefix} [Processing] {file_path.name}...")
                
                res_originals[uid] = load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=prefix)
                gc.collect()
            except Exception as e:
                print(f"\nError analyzing original {uid} ({method}): {e}")
                res_originals[uid] = None
        print(f"\n  Originals loaded.")

        res_covers = {}
        for i, uid in enumerate(common_ids, 1):
            try:
                file_path = cover_files[uid]
                prefix = f"  [{i}/{total_p}] ({i/total_p:.1%}) [Cover] [{method}]"
                cache_dir_method = cache_dir / method
                tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
                if tiny_path.exists():
                    print(f"{prefix} [Cache] {file_path.name}")
                else:
                    print(f"{prefix} [Processing] {file_path.name}...")
                
                res_covers[uid] = load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=prefix)
                gc.collect()
            except Exception as e:
                print(f"\nError analyzing cover {uid} ({method}): {e}")
                res_covers[uid] = None
        print(f"\n  Covers loaded.")

        lcs_list, dtw_list, mrr_sum, top5_hits, valid_count = [], [], 0.0, 0, 0
        ranks_list = []
        best_lcs, best_uid = -1.0, None
        detailed_results = []
        
        # Store results one by one to shuffle thresholds
        pairwise_lcs = []
        pairwise_lev = []
        pairwise_pitch_hist = []
        pairwise_dtw = []
        all_comparisons = []

        # Load comparisons cache if it exists
        comp_cache_path = cache_dir / method / f"comparison_cache_{dataset_dir.name}.json"
        comp_cache = {}
        comp_cache_changed = False
        if comp_cache_path.exists():
            try:
                with open(comp_cache_path, 'r', encoding='utf-8') as f:
                    comp_cache = json.load(f)
                print(f"  [Cache] Loaded previous comparisons from {comp_cache_path.name}")
            except Exception as e:
                print(f"  [Cache] Warning loading comparisons cache: {e}")
        
        for i, uid_cover in enumerate(common_ids, 1):
            try:
                print(f"  [{i}/{total_p}] ({i/total_p:.1%}) Comparing cover: ID {uid_cover}...", end='\r')
                if res_covers[uid_cover] is None: continue
                
                seq_cover = res_covers[uid_cover]['seq']
                pitch_m = res_covers[uid_cover]['pitch_midi']
                
                if pitch_m is None or len(pitch_m) == 0: continue
                
                f0_cover = np.nan_to_num(np.where(pitch_m > 0, 440.0 * np.power(2.0, (pitch_m - 69.0) / 12.0), 0))
                
                similarities = []
                for uid_orig in common_ids:
                    if res_originals[uid_orig] is None: continue
                    seq_orig = res_originals[uid_orig]['seq']
                    pitch_o = res_originals[uid_orig]['pitch_midi']
                    
                    # Generate a unique key and validation hash
                    key = f"{orig_files[uid_orig].name}:::{cover_files[uid_cover].name}"
                    cached_entry = comp_cache.get(key, {})
                    
                    import hashlib
                    orig_repr = ",".join(seq_orig) + f"|len:{len(pitch_o)}"
                    cover_repr = ",".join(seq_cover) + f"|len:{len(pitch_m)}"
                    h = hashlib.md5(f"{orig_repr}:::{cover_repr}".encode('utf-8')).hexdigest()
                    
                    # Invalidate if underlying data changed
                    if cached_entry.get("hash") != h:
                        cached_entry = {"hash": h}
                        
                    cache_updated = False
                    
                    # LCS
                    if "lcs_similarity" in cached_entry:
                        lcs_sim = cached_entry["lcs_similarity"]
                    else:
                        lcs_sim = calculate_lcs(seq_orig, seq_cover)
                        cached_entry["lcs_similarity"] = lcs_sim
                        cache_updated = True
                        
                    # Levenshtein
                    if "levenshtein_similarity" in cached_entry:
                        lev_sim = cached_entry["levenshtein_similarity"]
                    else:
                        lev_sim = calculate_levenshtein_similarity(seq_orig, seq_cover)
                        cached_entry["levenshtein_similarity"] = lev_sim
                        cache_updated = True
                        
                    # Pitch Histogram
                    if "pitch_hist_similarity" in cached_entry:
                        pitch_hist_sim = cached_entry["pitch_hist_similarity"]
                    else:
                        pitch_hist_sim = calculate_pitch_histogram_similarity(pitch_o, pitch_m)
                        cached_entry["pitch_hist_similarity"] = pitch_hist_sim
                        cache_updated = True
                        
                    # DTW
                    dtw_val = -1.0
                    is_correct = (uid_orig == uid_cover)
                    if args.dtw_all_pairs or is_correct:
                        if "dtw_distance" in cached_entry and cached_entry["dtw_distance"] != "":
                            dtw_val = cached_entry["dtw_distance"]
                        else:
                            f0_orig = np.nan_to_num(np.where(pitch_o > 0, 440.0 * np.power(2.0, (pitch_o - 69.0) / 12.0), 0))
                            try:
                                # Downsample ONLY on extremely long songs (more than 15 minutes / 38760 frames)
                                if len(f0_cover) > 38760:
                                    ds_factor = len(f0_cover) // 1000
                                    f0_cover_ds = f0_cover[::ds_factor]
                                    f0_orig_ds = f0_orig[::ds_factor]
                                else:
                                    f0_cover_ds = f0_cover
                                    f0_orig_ds = f0_orig
                                    
                                D, wp = librosa.sequence.dtw(f0_cover_ds.reshape(1, -1), f0_orig_ds.reshape(1, -1))
                                dtw_val = D[-1, -1] / len(wp)
                            except:
                                dtw_val = -1.0
                            cached_entry["dtw_distance"] = dtw_val if dtw_val >= 0 else ""
                            cache_updated = True
                            
                    if cache_updated:
                        comp_cache[key] = cached_entry
                        comp_cache_changed = True
                            
                    pairwise_lcs.append((lcs_sim, is_correct))
                    pairwise_lev.append((lev_sim, is_correct))
                    pairwise_pitch_hist.append((pitch_hist_sim, is_correct))
                    if dtw_val >= 0:
                        pairwise_dtw.append((dtw_val, is_correct))
                        
                    all_comparisons.append({
                        "cover_id": uid_cover,
                        "original_id": uid_orig,
                        "lcs_similarity": lcs_sim,
                        "levenshtein_similarity": lev_sim,
                        "pitch_hist_similarity": pitch_hist_sim,
                        "dtw_distance": dtw_val if dtw_val >= 0 else "",
                        "is_correct": 1 if is_correct else 0
                    })
                    
                    similarities.append((lcs_sim, uid_orig))
                
                if not similarities: continue
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                rank, true_sim = -1, 0.0
                for idx, (sim, r_uid) in enumerate(similarities):
                    if r_uid == uid_cover:
                        rank, true_sim = idx + 1, sim
                        break
                
                if rank != -1:
                    valid_count += 1
                    ranks_list.append(rank)
                    mrr_sum += 1.0 / rank
                    if rank <= 5: top5_hits += 1
                    lcs_list.append(true_sim)
                    
                    if true_sim > best_lcs:
                        best_lcs, best_uid = true_sim, uid_cover
                    
                    correct_dtw = -1.0
                    for comp in all_comparisons:
                        if comp["cover_id"] == uid_cover and comp["original_id"] == uid_cover:
                            if comp["dtw_distance"] != "":
                                correct_dtw = comp["dtw_distance"]
                            break
                    if correct_dtw >= 0:
                        dtw_list.append(correct_dtw)
                    
                    id_label = f"ID {uid_cover:02d}" if isinstance(uid_cover, int) else f"ID {uid_cover}"
                    detailed_results.append(f"{id_label} | LCS: {true_sim:.4f} | Rank: {rank:2d} | DTW: {correct_dtw:.4f}")
            except Exception as e:
                print(f"\nError processing cover {uid_cover} ({method}): {e}")
            finally:
                if 'res_cover' in locals():
                    del res_cover
                if i % 10 == 0:
                    gc.collect()
                    
        # Save comparisons cache if there were changes
        if comp_cache_changed:
            try:
                comp_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(comp_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(comp_cache, f, indent=2)
                print(f"\n  [Cache] Comparisons saved/updated at {comp_cache_path.name}")
            except Exception as e:
                print(f"\n  [Cache] Warning saving comparisons cache: {e}")

        # Metrics summary
        print(f"\n[{method}] Finished.")
        avg_lcs = np.mean(lcs_list) if lcs_list else 0
        mr = np.mean(ranks_list) if ranks_list else 0
        mrr = mrr_sum / valid_count if valid_count else 0
        mdr = np.median(ranks_list) if ranks_list else 0
        map_val = np.mean([1.0 / r for r in ranks_list]) if ranks_list else 0
        top5_prec = top5_hits / valid_count if valid_count else 0
        avg_dtw = np.mean(dtw_list) if dtw_list else 0
        
        print(f"[{method}] Results | LCS: {avg_lcs:.4f} | MR: {mr:.2f} | MRR: {mrr:.4f} | MDR: {mdr:.1f} | MAP: {map_val:.4f} | Top5: {top5_prec:.2%} | DTW: {avg_dtw:.4f}")
        
        # Export to CSV summary
        with open(summary_path, 'a') as f:
            f.write(f"{method},{valid_count},{avg_lcs:.6f},{mr:.6f},{mrr:.6f},{mdr:.1f},{map_val:.6f},{top5_prec:.6f},{avg_dtw:.6f}\n")
            
        # Evaluate binary classification and optimal thresholds
        best_thresh_lcs, best_metrics_lcs, curves_lcs = evaluate_binary_classification(pairwise_lcs, "LCS")
        best_thresh_lev, best_metrics_lev, curves_lev = evaluate_binary_classification(pairwise_lev, "Levenshtein")
        best_thresh_ph, best_metrics_ph, curves_ph = evaluate_binary_classification(pairwise_pitch_hist, "Pitch Histogram")
        best_thresh_dtw, best_metrics_dtw, curves_dtw = evaluate_binary_classification(pairwise_dtw, "DTW", lower_is_better=True)
        
        # Export all_comparisons.csv
        comp_csv_path = out_method_dir / "all_comparisons.csv"
        with open(comp_csv_path, 'w') as f:
            f.write("cover_id,original_id,lcs_similarity,levenshtein_similarity,pitch_hist_similarity,dtw_distance,is_correct\n")
            for comp in all_comparisons:
                f.write(f"{comp['cover_id']},{comp['original_id']},{comp['lcs_similarity']:.6f},{comp['levenshtein_similarity']:.6f},{comp['pitch_hist_similarity']:.6f},{comp['dtw_distance']},{comp['is_correct']}\n")
                
        # Export threshold curves
        for m_name, curves in [("lcs", curves_lcs), ("levenshtein", curves_lev), ("pitch_hist", curves_ph), ("dtw", curves_dtw)]:
            if not curves: continue
            curve_csv_path = out_method_dir / f"threshold_analysis_{m_name}.csv"
            with open(curve_csv_path, 'w') as f:
                f.write("threshold,tp,fp,fn,tn,precision,recall,f1_score,accuracy\n")
                for c in curves:
                    f.write(f"{c['threshold']:.4f},{c['tp']},{c['fp']},{c['fn']},{c['tn']},{c['precision']:.6f},{c['recall']:.6f},{c['f1_score']:.6f},{c['accuracy']:.6f}\n")
        
        # Export to Detailed TXT Report
        report_path = out_method_dir / "detailed_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"DETAILED REPORT - METHOD: {method}\n")
            f.write("="*50 + "\n")
            if best_uid is not None:
                f.write(f"IMAGES GENERATED FOR THE BEST MATCH (LCS = {best_lcs:.4f}):\n")
                f.write(f"  ID: {best_uid}\n")
                f.write(f"  Original: {orig_files[best_uid].name}\n")
                f.write(f"  Cover:    {cover_files[best_uid].name}\n")
                f.write("="*50 + "\n")
            f.write("\n".join(detailed_results) + "\n")
            f.write("="*50 + "\n")
            f.write(f"GENERAL SUMMARY:\n")
            f.write(f"Pairs evaluated: {valid_count}\n")
            f.write(f"Average LCS:    {avg_lcs:.4f}\n")
            f.write(f"Mean Rank:      {mr:.4f}\n")
            f.write(f"MRR:             {mrr:.4f}\n")
            f.write(f"Median Rank:     {mdr:.1f}\n")
            f.write(f"MAP:             {map_val:.4f}\n")
            f.write(f"Top-5 Precision: {top5_prec:.2%}\n")
            f.write(f"Average DTW:    {avg_dtw:.4f}\n")
            f.write("="*50 + "\n")
            f.write(f"BINARY CLASSIFICATION THRESHOLDS ANALYSIS (OPTIMIZING F1-SCORE):\n\n")
            
            for m_name, best_t, best_m in [
                ("LCS (Longest Common Subsequence)", best_thresh_lcs, best_metrics_lcs),
                ("Levenshtein (Edit Distance)", best_thresh_lev, best_metrics_lev),
                ("Pitch Class Histogram (Chroma Cosine)", best_thresh_ph, best_metrics_ph),
                ("DTW Distance (Optimal Path)", best_thresh_dtw, best_metrics_dtw)
            ]:
                f.write(f"--- Metric: {m_name} ---\n")
                if best_m:
                    f.write(f"  Optimal Threshold:  {best_t:.4f}\n")
                    f.write(f"  F1-Score:       {best_m['f1_score']:.4f}\n")
                    f.write(f"  Precision:      {best_m['precision']:.4f}\n")
                    f.write(f"  Recall (Sens.): {best_m['recall']:.4f}\n")
                    f.write(f"  Accuracy:       {best_m['accuracy']:.4f}\n")
                    f.write(f"  Confusion Matrix:\n")
                    f.write(f"    - TP (True Pos.):  {best_m['tp']}\n")
                    f.write(f"    - FP (False Pos.): {best_m['fp']}\n")
                    f.write(f"    - FN (False Neg.): {best_m['fn']}\n")
                    f.write(f"    - TN (True Neg.):  {best_m['tn']}\n")
                else:
                    f.write(f"  Not enough data to evaluate.\n")
                f.write("\n")
            f.write("="*50 + "\n")

        # Qualitative Plots for the absolute best match of this method
        if best_uid is not None:
            print(f"\n[{method}] Generating final qualitative plots for Best Match (ID {best_uid}, LCS={best_lcs:.4f})...")
            try:
                # Import visualization utilities
                from src.melody_analysis_v2.visualization import (
                    plot_melspectrogram, plot_melody_contour, plot_melody_only,
                    plot_energy_only, plot_melody_and_energy
                )
                
                # Process Original and Cover sequentially to save RAM
                print(f"  Processing Original (ID {best_uid})...")
                res_orig_best = analyzer.analyze_file(str(orig_files[best_uid]))
                
                # Novelty
                plot_boundary_detection(res_orig_best, output_path=out_method_dir / "fig_novelty_orig.pdf")
                plot_boundary_detection(res_orig_best, output_path=out_method_dir / "fig_novelty_orig.png")
                
                # SSM
                if res_orig_best.self_similarity is not None:
                    plot_self_similarity(res_orig_best, output_path=out_method_dir / "fig_ssm_orig.pdf")
                    plot_self_similarity(res_orig_best, output_path=out_method_dir / "fig_ssm_orig.png")
                
                # Melodic contour
                plot_melody_contour(res_orig_best, output_path=out_method_dir / "fig_contour_orig.pdf")
                plot_melody_contour(res_orig_best, output_path=out_method_dir / "fig_contour_orig.png")
                
                # Melodic contour only (no segments, no energy)
                plot_melody_only(res_orig_best, output_path=out_method_dir / "fig_contour_only_orig.pdf", show_segments=False)
                plot_melody_only(res_orig_best, output_path=out_method_dir / "fig_contour_only_orig.png", show_segments=False)
                
                # Energy only
                plot_energy_only(res_orig_best, output_path=out_method_dir / "fig_energy_only_orig.pdf")
                plot_energy_only(res_orig_best, output_path=out_method_dir / "fig_energy_only_orig.png")
                
                # Contour and Energy (no segments)
                plot_melody_and_energy(res_orig_best, output_path=out_method_dir / "fig_contour_and_energy_orig.pdf")
                plot_melody_and_energy(res_orig_best, output_path=out_method_dir / "fig_contour_and_energy_orig.png")
                
                # Spectrogram and Mel-spectrogram Orig
                try:
                    audio_plot, sr_plot = librosa.load(orig_files[best_uid], sr=analyzer.sample_rate)
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_orig_best, output_path=out_method_dir / f"fig_spectrogram_orig.pdf")
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_orig_best, output_path=out_method_dir / f"fig_spectrogram_orig.png")
                    
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_orig.pdf")
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_orig.png")
                    del audio_plot
                except Exception as spec_err:
                    print(f"  Warning plotting spectrogram/mel-spectrogram (orig): {spec_err}")
                
                plt.close('all')
                
                # Process Cover
                print(f"  Processing Cover (ID {best_uid})...")
                res_cover_best = analyzer.analyze_file(str(cover_files[best_uid]))
                
                # Novelty
                plot_boundary_detection(res_cover_best, output_path=out_method_dir / "fig_novelty_cover.pdf")
                plot_boundary_detection(res_cover_best, output_path=out_method_dir / "fig_novelty_cover.png")
                
                # SSM
                if res_cover_best.self_similarity is not None:
                    plot_self_similarity(res_cover_best, output_path=out_method_dir / "fig_ssm_cover.pdf")
                    plot_self_similarity(res_cover_best, output_path=out_method_dir / "fig_ssm_cover.png")
                
                # Melodic contour
                plot_melody_contour(res_cover_best, output_path=out_method_dir / "fig_contour_cover.pdf")
                plot_melody_contour(res_cover_best, output_path=out_method_dir / "fig_contour_cover.png")
                
                # Melodic contour only (no segments, no energy)
                plot_melody_only(res_cover_best, output_path=out_method_dir / "fig_contour_only_cover.pdf", show_segments=False)
                plot_melody_only(res_cover_best, output_path=out_method_dir / "fig_contour_only_cover.png", show_segments=False)
                
                # Energy only
                plot_energy_only(res_cover_best, output_path=out_method_dir / "fig_energy_only_cover.pdf")
                plot_energy_only(res_cover_best, output_path=out_method_dir / "fig_energy_only_cover.png")
                
                # Contour and Energy (no segments)
                plot_melody_and_energy(res_cover_best, output_path=out_method_dir / "fig_contour_and_energy_cover.pdf")
                plot_melody_and_energy(res_cover_best, output_path=out_method_dir / "fig_contour_and_energy_cover.png")
                
                # Spectrogram and Mel-spectrogram Cover
                try:
                    audio_plot, sr_plot = librosa.load(cover_files[best_uid], sr=analyzer.sample_rate)
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_cover_best, output_path=out_method_dir / f"fig_spectrogram_cover.pdf")
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_cover_best, output_path=out_method_dir / f"fig_spectrogram_cover.png")
                    
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_cover.pdf")
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_cover.png")
                    del audio_plot
                except Exception as spec_err:
                    print(f"  Warning plotting spectrogram/mel-spectrogram (cover): {spec_err}")
                
                plt.close('all')
                
                # Shared plots (Bands and Contour)
                plot_caplin_bands(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_bands.pdf")
                plot_caplin_contour(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour.pdf")
                plot_contour_only_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour_only.pdf")
                plot_energy_only_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_energy_only.pdf")
                plot_melody_and_energy_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour_and_energy.pdf")
                
                # --- NEW: Export the 9 diagram steps for the Best Match ---
                print(f"  Exporting 9 diagram steps for Best Match (ID {best_uid})...")
                exporter_orig = DiagramExporter(out_method_dir / "diagrama_pasos_original")
                exporter_orig.export_all(str(orig_files[best_uid]), method=method)
                
                exporter_cover = DiagramExporter(out_method_dir / "diagrama_pasos_cover")
                exporter_cover.export_all(str(cover_files[best_uid]), method=method)
                
                print(f"  Final plots generated successfully in {out_method_dir}")
                
                # Cleanup
                del res_orig_best, res_cover_best
                gc.collect()
            except Exception as plot_err:
                print(f"  Error generating final plots: {plot_err}")
            finally:
                plt.close('all')
                gc.collect()


if __name__ == "__main__":
    main()