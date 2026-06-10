import os
import re
import json
import argparse
import gc
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from src.melody_analysis_v2 import MelodyAnalyzer, MelodyClassifierPaper, MelodyFeatures, MelodySegmentAnnotation, DiagramExporter
from src.melody_analysis_v2.classifier_paper import calculate_lcs
from src.melody_analysis_v2.segmenter import MelodySegment
# pyrefly: ignore [missing-import]
from src.melody_analysis_v2.pipeline import MelodyAnalysisResult
from src.melody_analysis_v2.visualization import (
    plot_boundary_detection, 
    plot_self_similarity, 
    plot_spectrogram_with_segments
)

METHOD_CLASSIFICATION = {
    'pyin': 'Frecuencia Fundamental (F0 Extractor)',
    'yin': 'Frecuencia Fundamental (F0 Extractor)',
    'crepe': 'Frecuencia Fundamental (F0 Extractor)',
    'rmvpe': 'Frecuencia Fundamental (F0 Extractor)',
    'spice': 'Frecuencia Fundamental (F0 Extractor)',
    'jdc': 'Frecuencia Fundamental (F0 Extractor)',
    'fcn_f0': 'Frecuencia Fundamental (F0 Extractor)',
    'melodia': 'Extractor de Melodía (Melody Extractor - Salamon)',
    'demucs_crepe': 'Extractor de Melodía (Melody Extractor - Demucs+CREPE)',
    'bs_roformer_rmvpe': 'Extractor de Melodía (Melody Extractor - Roformer+RMVPE)',
    'bs_roformer': 'Extractor de Melodía (Melody Extractor - Roformer+RMVPE)',
    'demucs': 'Extractor de Melodía (Melody Extractor - Demucs+CREPE)',
    'bs_roformer_crepe': 'Extractor de Melodía (Melody Extractor - Roformer+CREPE)',
    'demucs_rmvpe': 'Extractor de Melodía (Melody Extractor - Demucs+RMVPE)',
    'basic_pitch': 'Extractor de Melodía (Melody Extractor - Spotify Basic Pitch)',
    'tachibana': 'Extractor de Melodía (Melody Extractor - Tachibana HPSS+Melodia)',
    'poliner': 'Extractor de Melodía (Melody Extractor - Poliner & Ellis STFT Peak)',
    'durrieu': 'Extractor de Melodía (Melody Extractor - Durrieu NMF+YIN)',
    'ensemble': 'Híbrido (Ensemble F0/Melodía)',
    'all_f0': 'Todos los extractores de F0',
    'all_melody': 'Todos los extractores de Melodía',
    'all': 'Todos los métodos'
}

def get_audio_files(directory_path: Path, match_mode: str = "id"):
    result = {}
    if not directory_path.exists():
        return result
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            if match_mode == "stem":
                # Normalizar el stem del archivo
                name = f.stem.lower()
                # Quitar sufijos comunes
                name = re.sub(r'[-_](cover|originales|original|orig|ref|covers|version|var)', '', name)
                # Quitar prefijos numéricos si están presentes
                name = re.sub(r'^\d+\s*[-_]?\s*', '', name)
                # Mantener solo caracteres alfanuméricos
                name = re.sub(r'[^a-z0-9]', '', name)
                key = name.strip()
                if key:
                    result[key] = f
            elif match_mode == "fuzzy":
                # Guardar por nombre completo (garantizado único)
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
    
    # 1. Agrupar por prefijo numérico (o grupo 0 por defecto)
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
            
    # 2. Emparejar para cada prefijo
    pair_id = 1
    
    def word_overlap(str1: str, str2: str) -> float:
        w1 = set(re.findall(r'[a-zA-Z0-9]+', str1.lower()))
        w2 = set(re.findall(r'[a-zA-Z0-9]+', str2.lower()))
        # Eliminar palabras de parada comunes
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
    """Calcula la similitud de Levenshtein normalizada entre dos secuencias de etiquetas."""
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
    """Calcula la similitud de coseno de los histogramas de clases de altura (croma)."""
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
    """Evalúa la clasificación binaria para una métrica dada sobre un rango de umbrales."""
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
            segments = [
                MelodySegmentAnnotation(
                    segment=MelodySegment(s["start_time"], s["end_time"], 0, 0),
                    label=s["label"],
                    confidence=s["confidence"],
                    descriptor=s["descriptor"]
                )
                for s in data["segments"]
            ]
            for s in segments:
                s.segment.start_index = int(np.searchsorted(features.times, s.segment.start_time))
                s.segment.end_index = int(np.searchsorted(features.times, s.segment.end_time))
                
            return MelodyAnalysisResult(features=features, segments=segments)
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
            
    # Si no existe tiny, usar el flujo en un subproceso para evitar leaks de memoria/OOM
    script_path = Path(__file__).parent / "src" / "melody_analysis_v2" / "analyze_single.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--file_path", str(file_path),
        "--method", method,
        "--cache_dir", str(cache_dir),
        "--label_prefix", label_prefix
    ]
    subprocess.run(cmd, check=True)
    
    # Una vez completado, el cache normal .json ya está guardado en disco.
    res = load_or_analyze(analyzer, file_path, method, cache_dir)
    seq = [s.label for s in res.segments]
    pitch_midi = res.features.pitch_midi.copy() if res.features.pitch_midi is not None else None
    
    # Escribir el tiny json
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


def parse_id3v2(filepath):
    """
    Parsea etiquetas ID3v2 de archivos MP3 en formato puro Python.
    Retorna un diccionario con 'title', 'artist'.
    """
    tags = {}
    try:
        with open(filepath, 'rb') as f:
            header = f.read(10)
            if len(header) < 10 or header[:3] != b'ID3':
                return tags
            
            major = header[3]
            size_bytes = header[6:10]
            tag_size = (size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3]
            
            tag_data = f.read(tag_size)
            
            offset = 0
            while offset + 10 < len(tag_data):
                if major == 2:
                    frame_id = tag_data[offset:offset+3].decode('latin1', errors='ignore')
                    if not frame_id or frame_id == '\x00\x00\x00':
                        break
                    frame_size = (tag_data[offset+3] << 16) | (tag_data[offset+4] << 8) | tag_data[offset+5]
                    frame_body = tag_data[offset+6:offset+6+frame_size]
                    offset += 6 + frame_size
                else:
                    frame_id = tag_data[offset:offset+4].decode('latin1', errors='ignore')
                    if not frame_id or frame_id == '\x00\x00\x00\x00':
                        break
                    fs_bytes = tag_data[offset+4:offset+8]
                    if major == 4:
                        frame_size = (fs_bytes[0] << 21) | (fs_bytes[1] << 14) | (fs_bytes[2] << 7) | fs_bytes[3]
                    else:
                        frame_size = (fs_bytes[0] << 24) | (fs_bytes[1] << 16) | (fs_bytes[2] << 8) | fs_bytes[3]
                    
                    frame_body = tag_data[offset+10:offset+10+frame_size]
                    offset += 10 + frame_size
                    
                if frame_id.startswith('T') and frame_id != 'TXXX':
                    if len(frame_body) > 1:
                        encoding = frame_body[0]
                        text_bytes = frame_body[1:]
                        try:
                            if encoding == 0:
                                text = text_bytes.decode('latin1', errors='ignore')
                            elif encoding == 1:
                                text = text_bytes.decode('utf-16', errors='ignore')
                            elif encoding == 2:
                                text = text_bytes.decode('utf-16-be', errors='ignore')
                            elif encoding == 3:
                                text = text_bytes.decode('utf-8', errors='ignore')
                            else:
                                text = text_bytes.decode('latin1', errors='ignore')
                        except Exception:
                            text = text_bytes.decode('latin1', errors='ignore')
                        
                        text = text.strip('\x00').strip()
                        
                        if frame_id in ['TPE1', 'TPE2', 'TP1']:
                            tags['artist'] = text
                        elif frame_id in ['TIT2', 'TT2']:
                            tags['title'] = text
    except Exception:
        pass
    return tags


def get_audio_metadata(filepath):
    filepath = Path(filepath)
    tags = parse_id3v2(filepath)
    title = tags.get('title')
    artist = tags.get('artist')
    if not title or not artist:
        filename = filepath.stem
        cleaned = re.sub(r'^\d+\s*[-_]?\s*', '', filename)
        parts = [p.strip() for p in cleaned.split(' - ')]
        if len(parts) >= 2:
            artist = artist or parts[0]
            title = title or ' - '.join(parts[1:])
        else:
            title = title or cleaned
            artist = artist or "Desconocido"
    return title.strip(), artist.strip()


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
            
            # Solo dibujar texto si el segmento es suficientemente ancho para evitar encimamiento
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
    
    # Save only as PNG
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
            
            # Label on top (solo si hay espacio)
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
    
    # Save only as PNG
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
    
    # Save only as PNG
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
    
    # Save only as PNG
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
    
    # Save only as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_dataset_comparative_table(dataset_dir: Path, output_dir: Path):
    summary_path = output_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        print(f"[Tabla Comparativa] No se encontró el archivo de resumen en {summary_path}")
        return
        
    dataset_name = dataset_dir.name
    
    # Read rows from summary_path and keep only the latest row per method
    method_rows = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    method_rows[parts[0]] = parts
    except Exception as e:
        print(f"Error leyendo {summary_path}: {e}")
        return
        
    if not method_rows:
        return
        
    # We will sort methods to keep the presentation consistent
    sorted_methods = sorted(list(method_rows.keys()))
    
    lines = []
    lines.append("-" * 90)
    lines.append(f"Dataset: {dataset_name}")
    lines.append("-" * 90)
    lines.append(f"{'Method':<25} | {'Avg. LCS (%)':<14} | {'MRR (%)':<10} | {'Top-5 (%)':<12} | {'Top-10 (%)':<12} | {'DTW':<10}")
    lines.append("-" * 90)
    
    for method in sorted_methods:
        row = method_rows[method]
        method_disp = method.upper()
        
        try:
            # Check length of the row to support both old 6-col and new 7-col formats
            if len(row) == 7:
                lcs = float(row[2]) * 100
                mrr = float(row[3]) * 100
                top5 = float(row[4]) * 100
                top10 = float(row[5]) * 100
                dtw = float(row[6])
                lines.append(f"{method_disp:<25} | {lcs:>12.2f}% | {mrr:>8.2f}% | {top5:>10.2f}% | {top10:>10.2f}% | {dtw:>10.2f}")
            else:
                # Old 6-column format (missing top10_prec)
                lcs = float(row[2]) * 100
                mrr = float(row[3]) * 100
                top5 = float(row[4]) * 100
                dtw = float(row[5])
                lines.append(f"{method_disp:<25} | {lcs:>12.2f}% | {mrr:>8.2f}% | {top5:>10.2f}% | {'-':>11} | {dtw:>10.2f}")
        except Exception as e:
            lines.append(f"{method.upper():<25} | Error al formatear fila: {e}")
            
    lines.append("-" * 90)
    
    table_content = "\n".join(lines) + "\n"
    table_path = dataset_dir / "tabla_comparativa.txt"
    try:
        table_path.write_text(table_content, encoding='utf-8')
        print(f"\n[Tabla Comparativa] Guardada exitosamente en {table_path}")
    except Exception as e:
        print(f"Error al escribir tabla comparativa: {e}")


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


def main():
    available_methods = [
        'all', 'all_f0', 'all_melody',
        'pyin', 'yin', 'crepe', 'rmvpe', 'spice', 'jdc', 'fcn_f0',
        'melodia', 'tachibana', 'poliner', 'durrieu', 'basic_pitch',
        'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
        'bs_roformer', 'demucs', 'ensemble'
    ]
    parser = argparse.ArgumentParser(description="Benchmark para extracción de melodía con soporte para caché.")
    parser.add_argument("--method", type=str, default=None, 
                        choices=available_methods,
                        help="Método de extracción a utilizar")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Directorio base del dataset")
    parser.add_argument("--orig_subdir", type=str, default="originales",
                        help="Subdirectorio de canciones originales")
    parser.add_argument("--cover_subdir", type=str, default="covers",
                        help="Subdirectorio de canciones covers")
    parser.add_argument("--output_dir", type=str, default="resultados_benchmark",
                        help="Directorio para salidas gráficas y reportes (si es relativo se resolverá dentro de la carpeta del dataset)")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directorio para la caché de análisis JSON")
    parser.add_argument("--match_mode", type=str, default=None,
                        choices=["id", "stem"],
                        help="Método de emparejamiento: 'id' (ID numérico de prefijo) o 'stem' (nombre/stem normalizado)")
    parser.add_argument("--dtw_all_pairs", action="store_true",
                        help="Calcular DTW para todos los pares (lento)")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Eliminar la caché existente para el método seleccionado antes de comenzar")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()

    if args.dataset_dir is None:
        datasets = find_available_datasets(base_dir)
        if not datasets:
            print("\nNo se detectaron carpetas de dataset automáticamente en el directorio base.")
            manual = input("Por favor ingrese la ruta o nombre del dataset a utilizar: ").strip()
            args.dataset_dir = manual
        else:
            print("\n=== Selección de Dataset ===")
            for i, d in enumerate(datasets, 1):
                print(f"{i}. {d}")
            print(f"{len(datasets) + 1}. [Procesar TODOS los datasets de una vez]")
            print(f"{len(datasets) + 2}. [Ingresar otra ruta manual...]")
            
            while True:
                try:
                    choice = input(f"\nSeleccione un dataset (1-{len(datasets) + 2}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(datasets):
                        args.dataset_dir = datasets[idx]
                        break
                    elif idx == len(datasets):
                        args.dataset_dir = "all"
                        break
                    elif idx == len(datasets) + 1:
                        manual = input("Ingrese la ruta o nombre del dataset: ").strip()
                        if manual:
                            args.dataset_dir = manual
                            break
                    else:
                        print(f"Error: Por favor seleccione un número entre 1 y {len(datasets) + 2}.")
                except ValueError:
                    if choice in datasets:
                        args.dataset_dir = choice
                        break
                    elif choice.lower() == "all":
                        args.dataset_dir = "all"
                        break
                    print("Error: Entrada no válida. Ingrese el número de opción o el nombre exacto.")

    if args.method is None:
        print("\n=== Selección de Método de Extracción ===")
        
        f0_methods = ['pyin', 'yin', 'crepe', 'rmvpe', 'spice', 'jdc', 'fcn_f0']
        melody_methods = [
            'poliner', 'durrieu', 'tachibana', 'melodia', 'basic_pitch',
            'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
            'bs_roformer', 'demucs'
        ]
        other_methods = ['ensemble']
        
        idx_map = {}
        curr_idx = 1
        
        print("\n--- Extractores de F0 (Frecuencia Fundamental) ---")
        for m in f0_methods:
            print(f"  {curr_idx:2d}. {m}")
            idx_map[curr_idx] = m
            curr_idx += 1
            
        print(f"  {curr_idx:2d}. {'all_f0':<20} [Todos los extractores de F0]")
        idx_map[curr_idx] = 'all_f0'
        curr_idx += 1
        
        print("\n--- Extractores de Melodía ---")
        for m in melody_methods:
            print(f"  {curr_idx:2d}. {m}")
            idx_map[curr_idx] = m
            curr_idx += 1
            
        print(f"  {curr_idx:2d}. {'all_melody':<20} [Todos los extractores de Melodía]")
        idx_map[curr_idx] = 'all_melody'
        curr_idx += 1
        
        print("\n--- Otros / Especiales ---")
        for m in other_methods:
            classification = METHOD_CLASSIFICATION.get(m, "")
            print(f"  {curr_idx:2d}. {m:<20} [{classification}]")
            idx_map[curr_idx] = m
            curr_idx += 1
            
        print(f"  {curr_idx:2d}. {'all':<20} [Todos los métodos]")
        idx_map[curr_idx] = 'all'
        
        while True:
            try:
                choice = input(f"\nSeleccione un método (1-{curr_idx}): ").strip()
                if choice.lower() in available_methods:
                    args.method = choice.lower()
                    break
                idx = int(choice)
                if 1 <= idx <= curr_idx:
                    args.method = idx_map[idx]
                    break
                else:
                    print(f"Error: Por favor seleccione un número entre 1 y {curr_idx}.")
                    gc.collect()
            except ValueError:
                if choice.strip().lower() in available_methods:
                    args.method = choice.strip().lower()
                    break
                print("Error: Entrada no válida. Ingrese el número del método o el nombre.")
                gc.collect()

    if args.match_mode is None:
        print("\n=== Selección de Método de Emparejamiento ===")
        print("1. Por ID Numérico (ej: '01 - Pedro Infante.wav' con '01 - Cover.mp3')")
        print("2. Por Nombre / Stem Exacto (ej: 'Te_Vi_Venir_Original.wav' con 'Te Vi Venir (Covers).mp3')")
        print("3. Emparejamiento Inteligente / Fuzzy (Para nombres complejos o música clásica, ej: '02 - Symphony No. 40...')")
        
        while True:
            choice = input("\nSeleccione el método de emparejamiento (1-3) [Por defecto: 1]: ").strip()
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
                print("Error: Por favor seleccione 1, 2 o 3.")
                
    args.match_by_stem = (args.match_mode == "stem")
    run_benchmark_execution(args, base_dir)

def run_single_dataset_benchmark(dataset_dir: Path, methods: list, args, base_dir: Path, cache_dir: Path):
    orig_dir = dataset_dir / args.orig_subdir
    cover_dir = dataset_dir / args.cover_subdir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = dataset_dir / output_dir
        
    if not orig_dir.exists() or not cover_dir.exists():
        print(f"Las carpetas de origen '{orig_dir}' y/o '{cover_dir}' no existen en {dataset_dir.name}. Saltando...")
        return
        
    orig_files = get_audio_files(orig_dir, match_mode=args.match_mode)
    cover_files = get_audio_files(cover_dir, match_mode=args.match_mode)
    
    if args.match_mode == "fuzzy":
        orig_files, cover_files = pair_files_fuzzy(orig_files, cover_files)
        common_ids = sorted(list(orig_files.keys()))
    else:
        common_ids = sorted(list(set(orig_files.keys()).intersection(set(cover_files.keys()))))
        
    print("\n" + "="*80)
    print(f" PROCESANDO DATASET: {dataset_dir.name} ({len(common_ids)} pares encontrados)")
    print("="*80)
    
    if not common_ids:
        print("No se encontraron pares válidos. Saltando...")
        return

    print("Métodos a evaluar:")
    for m in methods:
        classification = METHOD_CLASSIFICATION.get(m, "Desconocido")
        print(f"  - {m}: {classification}")

    classifier = MelodyClassifierPaper()
    summary_path = output_dir / "benchmark_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_path.exists():
        with open(summary_path, 'w') as f:
            f.write("metodo,pares,lcs_promedio,mrr,top5_prec,top10_prec,dtw_promedio\n")

    for method in methods:
        classification = METHOD_CLASSIFICATION.get(method, "Desconocido")
        print(f"\n[{method}] Procesando... ({classification})")
        analyzer = MelodyAnalyzer(extraction_method=method, classifier=classifier)
        
        out_method_dir = output_dir / method
        out_method_dir.mkdir(parents=True, exist_ok=True)

        res_originals = {}
        total_p = len(common_ids)
        for i, uid in enumerate(common_ids, 1):
            try:
                file_path = orig_files[uid]
                prefix = f"  [{i}/{total_p}] ({i/total_p:.1%}) [Original]"
                cache_dir_method = cache_dir / method
                tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
                if tiny_path.exists():
                    print(f"{prefix} [Caché] {file_path.name}")
                else:
                    print(f"{prefix} [Procesando] {file_path.name}...")
                
                res_originals[uid] = load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=prefix)
                gc.collect()
            except Exception as e:
                print(f"\nError analizando original {uid} ({method}): {e}")
                res_originals[uid] = None
        print(f"\n  Originales cargados.")

        res_covers = {}
        for i, uid in enumerate(common_ids, 1):
            try:
                file_path = cover_files[uid]
                prefix = f"  [{i}/{total_p}] ({i/total_p:.1%}) [Cover]"
                cache_dir_method = cache_dir / method
                tiny_path = cache_dir_method / f"{file_path.stem}.tiny.json"
                if tiny_path.exists():
                    print(f"{prefix} [Caché] {file_path.name}")
                else:
                    print(f"{prefix} [Procesando] {file_path.name}...")
                
                res_covers[uid] = load_or_analyze_light(analyzer, file_path, method, cache_dir, label_prefix=prefix)
                gc.collect()
            except Exception as e:
                print(f"\nError analizando cover {uid} ({method}): {e}")
                res_covers[uid] = None
        print(f"\n  Covers cargados.")
        lcs_list, dtw_list, mrr_sum, top5_hits, top10_hits, valid_count = [], [], 0.0, 0, 0, 0
        best_lcs, best_uid = -1.0, None
        detailed_results = []
        
        # Almacenar resultados uno a uno para barajar umbrales
        pairwise_lcs = []
        pairwise_lev = []
        pairwise_pitch_hist = []
        pairwise_dtw = []
        all_comparisons = []

        # Cargar caché de comparaciones si existe
        comp_cache_path = cache_dir / method / f"comparison_cache_{dataset_dir.name}.json"
        comp_cache = {}
        comp_cache_changed = False
        if comp_cache_path.exists():
            try:
                with open(comp_cache_path, 'r', encoding='utf-8') as f:
                    comp_cache = json.load(f)
                print(f"  [Caché] Cargadas comparaciones previas desde {comp_cache_path.name}")
            except Exception as e:
                print(f"  [Caché] Advertencia al cargar caché de comparaciones: {e}")
        
        for i, uid_cover in enumerate(common_ids, 1):
            try:
                print(f"  [{i}/{total_p}] ({i/total_p:.1%}) Comparando cover: ID {uid_cover}...", end='\r')
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
                    
                    # Generar una clave única y hash de validación
                    key = f"{orig_files[uid_orig].name}:::{cover_files[uid_cover].name}"
                    cached_entry = comp_cache.get(key, {})
                    
                    import hashlib
                    orig_repr = ",".join(seq_orig) + f"|len:{len(pitch_o)}"
                    cover_repr = ",".join(seq_cover) + f"|len:{len(pitch_m)}"
                    h = hashlib.md5(f"{orig_repr}:::{cover_repr}".encode('utf-8')).hexdigest()
                    
                    # Invalida si los datos subyacentes cambiaron
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
                                # Submuestrear SOLO en canciones extremadamente largas (más de 15 minutos / 38760 tramas)
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
                    mrr_sum += 1.0 / rank
                    if rank <= 5: top5_hits += 1
                    if rank <= 10: top10_hits += 1
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
                print(f"\nError procesando cover {uid_cover} ({method}): {e}")
            finally:
                if 'res_cover' in locals():
                    del res_cover
                if i % 10 == 0:
                    gc.collect()
                    
        # Guardar caché de comparaciones si hubo cambios
        if comp_cache_changed:
            try:
                comp_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(comp_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(comp_cache, f, indent=2)
                print(f"\n  [Caché] Comparaciones guardadas/actualizadas en {comp_cache_path.name}")
            except Exception as e:
                print(f"\n  [Caché] Advertencia al guardar caché de comparaciones: {e}")

        # Metrics summary
        print(f"\n[{method}] Finalizado.")
        avg_lcs = np.mean(lcs_list) if lcs_list else 0
        mrr = mrr_sum / valid_count if valid_count else 0
        top5_prec = top5_hits / valid_count if valid_count else 0
        top10_prec = top10_hits / valid_count if valid_count else 0
        avg_dtw = np.mean(dtw_list) if dtw_list else 0
        
        print(f"[{method}] Resultados | LCS: {avg_lcs:.4f} | MRR: {mrr:.4f} | Top5: {top5_prec:.2%} | Top10: {top10_prec:.2%} | DTW: {avg_dtw:.4f}")
        
        # Export to CSV summary
        with open(summary_path, 'a') as f:
            f.write(f"{method},{valid_count},{avg_lcs:.6f},{mrr:.6f},{top5_prec:.6f},{top10_prec:.6f},{avg_dtw:.6f}\n")
            
        # Evaluar clasificación binaria y umbrales óptimos
        best_thresh_lcs, best_metrics_lcs, curves_lcs = evaluate_binary_classification(pairwise_lcs, "LCS")
        best_thresh_lev, best_metrics_lev, curves_lev = evaluate_binary_classification(pairwise_lev, "Levenshtein")
        best_thresh_ph, best_metrics_ph, curves_ph = evaluate_binary_classification(pairwise_pitch_hist, "Pitch Histogram")
        best_thresh_dtw, best_metrics_dtw, curves_dtw = evaluate_binary_classification(pairwise_dtw, "DTW", lower_is_better=True)
        
        # Exportar comparativas_todas.csv
        comp_csv_path = out_method_dir / "comparativas_todas.csv"
        with open(comp_csv_path, 'w') as f:
            f.write("cover_id,original_id,lcs_similarity,levenshtein_similarity,pitch_hist_similarity,dtw_distance,is_correct\n")
            for comp in all_comparisons:
                f.write(f"{comp['cover_id']},{comp['original_id']},{comp['lcs_similarity']:.6f},{comp['levenshtein_similarity']:.6f},{comp['pitch_hist_similarity']:.6f},{comp['dtw_distance']},{comp['is_correct']}\n")
                
        # Exportar curvas de umbrales
        for m_name, curves in [("lcs", curves_lcs), ("levenshtein", curves_lev), ("pitch_hist", curves_ph), ("dtw", curves_dtw)]:
            if not curves: continue
            curve_csv_path = out_method_dir / f"analisis_umbrales_{m_name}.csv"
            with open(curve_csv_path, 'w') as f:
                f.write("threshold,tp,fp,fn,tn,precision,recall,f1_score,accuracy\n")
                for c in curves:
                    f.write(f"{c['threshold']:.4f},{c['tp']},{c['fp']},{c['fn']},{c['tn']},{c['precision']:.6f},{c['recall']:.6f},{c['f1_score']:.6f},{c['accuracy']:.6f}\n")
        
        # Export to Detailed TXT Report
        report_path = out_method_dir / "reporte_detallado.txt"
        with open(report_path, 'w') as f:
            f.write(f"REPORTE DETALLADO - METODO: {method}\n")
            f.write("="*50 + "\n")
            if best_uid is not None:
                f.write(f"IMAGENES GENERADAS PARA EL MEJOR MATCH (LCS = {best_lcs:.4f}):\n")
                f.write(f"  ID: {best_uid}\n")
                f.write(f"  Original: {orig_files[best_uid].name}\n")
                f.write(f"  Cover:    {cover_files[best_uid].name}\n")
                f.write("="*50 + "\n")
            f.write("\n".join(detailed_results) + "\n")
            f.write("="*50 + "\n")
            f.write(f"RESUMEN GENERAL:\n")
            f.write(f"Pares evaluados: {valid_count}\n")
            f.write(f"LCS Promedio:    {avg_lcs:.4f}\n")
            f.write(f"MRR:             {mrr:.4f}\n")
            f.write(f"Top-5 Precision: {top5_prec:.2%}\n")
            f.write(f"Top-10 Precision: {top10_prec:.2%}\n")
            f.write(f"DTW Promedio:    {avg_dtw:.4f}\n")
            f.write("="*50 + "\n")
            f.write(f"ANALISIS DE UMBRALES DE CLASIFICACION BINARIA (OPTIMIZANDO F1-SCORE):\n\n")
            
            for m_name, best_t, best_m in [
                ("LCS (Subsecuencia Comun)", best_thresh_lcs, best_metrics_lcs),
                ("Levenshtein (Distancia Edicion)", best_thresh_lev, best_metrics_lev),
                ("Pitch Class Histogram (Croma Coseno)", best_thresh_ph, best_metrics_ph),
                ("DTW Distance (Camino Optimo)", best_thresh_dtw, best_metrics_dtw)
            ]:
                f.write(f"--- Metrica: {m_name} ---\n")
                if best_m:
                    f.write(f"  Umbral Optimo:  {best_t:.4f}\n")
                    f.write(f"  F1-Score:       {best_m['f1_score']:.4f}\n")
                    f.write(f"  Precision:      {best_m['precision']:.4f}\n")
                    f.write(f"  Recall (Sens.): {best_m['recall']:.4f}\n")
                    f.write(f"  Accuracy:       {best_m['accuracy']:.4f}\n")
                    f.write(f"  Matriz de Confusion:\n")
                    f.write(f"    - TP (True Pos.):  {best_m['tp']}\n")
                    f.write(f"    - FP (False Pos.): {best_m['fp']}\n")
                    f.write(f"    - FN (False Neg.): {best_m['fn']}\n")
                    f.write(f"    - TN (True Neg.):  {best_m['tn']}\n")
                else:
                    f.write(f"  Sin datos suficientes para evaluar.\n")
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
                
                # Extract metadata
                meta_orig = get_audio_metadata(orig_files[best_uid])
                meta_cover = get_audio_metadata(cover_files[best_uid])
                title_orig = f"{meta_orig[0]} ({meta_orig[1]})"
                title_cover = f"{meta_cover[0]} ({meta_cover[1]})"
                
                # Process Original and Cover sequentially to save RAM
                print(f"  Processing Original (ID {best_uid})...")
                res_orig_best = analyzer.analyze_file(str(orig_files[best_uid]))
                
                # Novelty
                plot_boundary_detection(res_orig_best, output_path=out_method_dir / "fig_novelty_orig.png", title=f"Boundary Detection (Original)\nSong: {title_orig}")
                
                # SSM
                if res_orig_best.self_similarity is not None:
                    plot_self_similarity(res_orig_best, output_path=out_method_dir / "fig_ssm_orig.png", title=f"SSM (Original)\nSong: {title_orig}")
                
                # Melodic contour
                plot_melody_contour(res_orig_best, output_path=out_method_dir / "fig_contour_orig.png", title=f"Melodic Contour (Original)\nSong: {title_orig}")
                
                # Melodic contour only (no segments, no energy)
                plot_melody_only(res_orig_best, output_path=out_method_dir / "fig_contour_only_orig.png", show_segments=False, title=f"Melodic Contour Only (Original)\nSong: {title_orig}")
                
                # Energy only
                plot_energy_only(res_orig_best, output_path=out_method_dir / "fig_energy_only_orig.png", title=f"Normalized Energy Only (Original)\nSong: {title_orig}")
                
                # Contour and Energy (no segments)
                plot_melody_and_energy(res_orig_best, output_path=out_method_dir / "fig_contour_and_energy_orig.png", title=f"Melodic Contour & Energy (Original)\nSong: {title_orig}")
                
                # Spectrogram and Mel-spectrogram Orig
                try:
                    audio_plot, sr_plot = librosa.load(orig_files[best_uid], sr=analyzer.sample_rate)
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_orig_best, output_path=out_method_dir / "fig_spectrogram_orig.png", title=f"Spectrogram with Segments (Original)\nSong: {title_orig}")
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_orig.png", title=f"Mel-spectrogram (Original)\nSong: {title_orig}")
                    del audio_plot
                except Exception as spec_err:
                    print(f"  Warning plotting spectrogram/mel-spectrogram (orig): {spec_err}")
                
                plt.close('all')
                
                # Process Cover
                print(f"  Processing Cover (ID {best_uid})...")
                res_cover_best = analyzer.analyze_file(str(cover_files[best_uid]))
                
                # Novelty
                plot_boundary_detection(res_cover_best, output_path=out_method_dir / "fig_novelty_cover.png", title=f"Boundary Detection (Cover)\nSong: {title_cover}")
                
                # SSM
                if res_cover_best.self_similarity is not None:
                    plot_self_similarity(res_cover_best, output_path=out_method_dir / "fig_ssm_cover.png", title=f"SSM (Cover)\nSong: {title_cover}")
                
                # Melodic contour
                plot_melody_contour(res_cover_best, output_path=out_method_dir / "fig_contour_cover.png", title=f"Melodic Contour (Cover)\nSong: {title_cover}")
                
                # Melodic contour only (no segments, no energy)
                plot_melody_only(res_cover_best, output_path=out_method_dir / "fig_contour_only_cover.png", show_segments=False, title=f"Melodic Contour Only (Cover)\nSong: {title_cover}")
                
                # Energy only
                plot_energy_only(res_cover_best, output_path=out_method_dir / "fig_energy_only_cover.png", title=f"Normalized Energy Only (Cover)\nSong: {title_cover}")
                
                # Contour and Energy (no segments)
                plot_melody_and_energy(res_cover_best, output_path=out_method_dir / "fig_contour_and_energy_cover.png", title=f"Melodic Contour & Energy (Cover)\nSong: {title_cover}")
                
                # Spectrogram and Mel-spectrogram Cover
                try:
                    audio_plot, sr_plot = librosa.load(cover_files[best_uid], sr=analyzer.sample_rate)
                    plot_spectrogram_with_segments(audio_plot, sr_plot, res_cover_best, output_path=out_method_dir / "fig_spectrogram_cover.png", title=f"Spectrogram with Segments (Cover)\nSong: {title_cover}")
                    plot_melspectrogram(audio_plot, sr_plot, output_path=out_method_dir / "fig_melspectrogram_cover.png", title=f"Mel-spectrogram (Cover)\nSong: {title_cover}")
                    del audio_plot
                except Exception as spec_err:
                    print(f"  Warning plotting spectrogram/mel-spectrogram (cover): {spec_err}")
                
                plt.close('all')
                
                # Shared plots (Bands and Contour)
                plot_caplin_bands(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_bands.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_caplin_contour(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_contour_only_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour_only.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_energy_only_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_energy_only.png", meta_orig=meta_orig, meta_cover=meta_cover)
                plot_melody_and_energy_comparison(res_orig_best, res_cover_best, out_method_dir / "fig_qualitative_contour_and_energy.png", meta_orig=meta_orig, meta_cover=meta_cover)
                
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

    # Guardar tabla comparativa consolidada en la carpeta del dataset
    save_dataset_comparative_table(dataset_dir, output_dir)


def run_benchmark_execution(args, base_dir):
    # Resolver rutas base compartidas
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir

    if args.method == 'all':
        methods = [
            'pyin', 'yin', 'crepe', 'ensemble',
            'poliner', 'durrieu', 'tachibana', 'melodia', 'basic_pitch',
            'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe'
        ]
    elif args.method == 'all_f0':
        methods = ['pyin', 'yin', 'crepe', 'rmvpe', 'spice', 'jdc', 'fcn_f0']
    elif args.method == 'all_melody':
        methods = [
            'poliner', 'durrieu', 'tachibana', 'melodia', 'basic_pitch',
            'demucs_crepe', 'bs_roformer_rmvpe', 'bs_roformer_crepe', 'demucs_rmvpe',
            'bs_roformer', 'demucs'
        ]
    else:
        methods = [args.method]

    # Limpieza de caché opcional (CLI o interactiva)
    if args.clear_cache:
        for m in methods:
            method_cache_dir = cache_dir / m
            if method_cache_dir.exists():
                print(f"[Caché] Eliminando archivos de caché existentes para el método '{m}' en {method_cache_dir}...")
                import shutil
                shutil.rmtree(method_cache_dir)
                print(f"[Caché] Eliminación completada para '{m}'.")
    else:
        any_cache_exists = any((cache_dir / m).exists() and (cache_dir / m).is_dir() and any((cache_dir / m).iterdir()) for m in methods)
        if any_cache_exists:
            ans = input(f"\n¿Desea eliminar la caché existente para los métodos a evaluar antes de comenzar? (s/n): ").strip().lower()
            if ans in ['s', 'si', 'y', 'yes']:
                for m in methods:
                    method_cache_dir = cache_dir / m
                    if method_cache_dir.exists():
                        print(f"[Caché] Eliminando archivos de caché en {method_cache_dir}...")
                        import shutil
                        shutil.rmtree(method_cache_dir)
                print("[Caché] Eliminación completada.")

    # Determinar datasets a procesar
    if args.dataset_dir == "all":
        datasets_to_process = []
        datasets_names = find_available_datasets(base_dir)
        for name in datasets_names:
            path = Path(name)
            if not path.is_absolute():
                path = base_dir / path
            datasets_to_process.append(path)
    else:
        path = Path(args.dataset_dir)
        if not path.is_absolute():
            path = base_dir / path
        datasets_to_process = [path]

    # Procesar datasets seleccionados
    for dataset_dir in datasets_to_process:
        try:
            run_single_dataset_benchmark(dataset_dir, methods, args, base_dir, cache_dir)
        except Exception as e:
            print(f"\nError al procesar el dataset '{dataset_dir.name}': {e}")


if __name__ == "__main__":
    main()