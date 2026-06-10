import argparse
import sys
import re
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from src.melody_analysis_v2 import MelodyAnalyzer, MelodyClassifierPaper
from src.melody_analysis_v2.visualization import (
    plot_boundary_detection, 
    plot_self_similarity, 
    plot_spectrogram_with_segments,
    plot_melody_contour,
    plot_melody_only,
    plot_energy_only,
    plot_melody_and_energy,
    plot_segment_extraction
)
from run_melody_benchmark import (
    get_audio_metadata,
    plot_caplin_bands,
    plot_caplin_contour,
    plot_contour_only_comparison,
    plot_energy_only_comparison,
    plot_melody_and_energy_comparison
)

def get_audio_files_by_id(directory_path: Path):
    result = {}
    for f in directory_path.iterdir():
        if f.is_file() and f.suffix.lower() in ['.mp3', '.wav']:
            match = re.search(r'^(\d+)', f.name)
            if match:
                file_id = int(match.group(1))
                result[file_id] = f
    return result

def main():
    parser = argparse.ArgumentParser(description="Generar visualizaciones completas para un par de canciones específico.")
    parser.add_argument("id", type=int, help="ID de la canción (ej: 01, 05, 80)")
    parser.add_argument("--method", type=str, default="crepe", help="Método de extracción (crepe, rmvpe, bs_roformer, etc.)")
    parser.add_argument("--output_dir", type=str, default="salidas_visualizacion", help="Directorio de salida")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.absolute()
    dataset_dir = base_dir / "dataset_clei"
    orig_dir = dataset_dir / "originales"
    cover_dir = dataset_dir / "covers"
    
    if not orig_dir.exists() or not cover_dir.exists():
        print("Error: No se encontraron las carpetas de dataset_clei.")
        return
        
    orig_files = get_audio_files_by_id(orig_dir)
    cover_files = get_audio_files_by_id(cover_dir)
    
    if args.id not in orig_files or args.id not in cover_files:
        print(f"Error: No se encontró el ID {args.id} en originales o covers.")
        return
        
    print(f"Procesando ID {args.id} con el método {args.method}...")
    
    classifier = MelodyClassifierPaper()
    analyzer = MelodyAnalyzer(extraction_method=args.method, classifier=classifier)
    
    out_dir = base_dir / args.output_dir / f"ID_{args.id:02d}_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_dict = {}
    meta_dict = {}
    for name, path in [("original", orig_files[args.id]), ("cover", cover_files[args.id])]:
        print(f"  Analizando {name}: {path.name}...")
        res = analyzer.analyze_file(str(path))
        res_dict[name] = res
        
        # Get metadata
        meta = get_audio_metadata(path)
        meta_dict[name] = meta
        title_str = f"{meta[0]} ({meta[1]})"
        
        # 1. Melodic Contour
        plot_melody_contour(res, output_path=out_dir / f"contour_{name}.png", title=f"Melodic Contour ({name.capitalize()})\nSong: {title_str}")
        
        # 1b. Melodic Contour Only (no segments, no energy)
        plot_melody_only(res, output_path=out_dir / f"contour_only_{name}.png", show_segments=False, title=f"Melodic Contour Only ({name.capitalize()})\nSong: {title_str}")
        
        # 1c. Energy Only
        plot_energy_only(res, output_path=out_dir / f"energy_only_{name}.png", title=f"Normalized Energy Only ({name.capitalize()})\nSong: {title_str}")

        # 1d. Contour and Energy
        plot_melody_and_energy(res, output_path=out_dir / f"contour_and_energy_{name}.png", title=f"Melodic Contour & Energy ({name.capitalize()})\nSong: {title_str}")
        
        # 2. Segments only (Bands)
        plot_segment_extraction(res, output_path=out_dir / f"bands_{name}.png", title=f"Melodic Segmentation Bands ({name.capitalize()})\nSong: {title_str}")
        
        # 3. Novelty Curves
        try:
            plot_boundary_detection(res, output_path=out_dir / f"novelty_{name}.png", title=f"Boundary Detection ({name.capitalize()})\nSong: {title_str}")
        except Exception as e:
            print(f"    Advertencia (Novedad): {e}")
            
        # 4. SSM
        try:
            plot_self_similarity(res, output_path=out_dir / f"ssm_{name}.png", title=f"SSM ({name.capitalize()})\nSong: {title_str}")
        except Exception as e:
            print(f"    Advertencia (SSM): {e}")
            
        # 5. Spectrogram
        try:
            audio, sr = librosa.load(path, sr=analyzer.sample_rate)
            plot_spectrogram_with_segments(audio, sr, res, output_path=out_dir / f"spectrogram_{name}.png", title=f"Spectrogram with Segments ({name.capitalize()})\nSong: {title_str}")
        except Exception as e:
            print(f"    Advertencia (Espectrograma): {e}")
            
    # Shared/Comparative plots
    print("  Generando gráficos comparativos...")
    try:
        plot_caplin_bands(res_dict["original"], res_dict["cover"], out_dir / "fig_qualitative_bands.png", meta_orig=meta_dict["original"], meta_cover=meta_dict["cover"])
        plot_caplin_contour(res_dict["original"], res_dict["cover"], out_dir / "fig_qualitative_contour.png", meta_orig=meta_dict["original"], meta_cover=meta_dict["cover"])
        plot_contour_only_comparison(res_dict["original"], res_dict["cover"], out_dir / "fig_qualitative_contour_only.png", meta_orig=meta_dict["original"], meta_cover=meta_dict["cover"])
        plot_energy_only_comparison(res_dict["original"], res_dict["cover"], out_dir / "fig_qualitative_energy_only.png", meta_orig=meta_dict["original"], meta_cover=meta_dict["cover"])
        plot_melody_and_energy_comparison(res_dict["original"], res_dict["cover"], out_dir / "fig_qualitative_contour_and_energy.png", meta_orig=meta_dict["original"], meta_cover=meta_dict["cover"])
    except Exception as e:
        print(f"  Advertencia (Gráficos Comparativos): {e}")
        
    print(f"\nVisualizaciones generadas en: {out_dir}")

if __name__ == "__main__":
    main()
