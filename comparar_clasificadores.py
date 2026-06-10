import sys
from pathlib import Path
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# Setup path
sys.path.append(str(Path(__file__).parent / "src"))

from melody_analysis_v2 import (
    MelodyAnalyzer, 
    MelodyClassifier, 
    MelodyClassifierPaper, 
    MelodyClassifierV1Rules,
    plot_melody_contour
)

def main():
    audio_path = "1.mp3"
    if not Path(audio_path).exists():
        print(f"Error: {audio_path} no encontrado.")
        return

    # 1. Configurar los 3 clasificadores
    c1 = MelodyClassifier()      # Original v2.5 (Caplin)
    c2 = MelodyClassifierPaper() # Paper (Q/A estricto)
    c3 = MelodyClassifierV1Rules() # Reglas v1.x (Portadas)

    clasificadores = [
        ("Caplin (Original 2.5)", c1),
        ("Paper (Pregunta/Respuesta)", c2),
        ("Reglas v1.x (Original MC-MSA)", c3)
    ]

    output_dir = Path("salidas_comparativa_clasificadores")
    output_dir.mkdir(exist_ok=True)

    # Analizar y graficar
    fig, axes = plt.subplots(len(clasificadores), 1, figsize=(15, 4 * len(clasificadores)), sharex=True)
    
    print(f"Iniciando comparativa para {audio_path}...")

    for i, (name, clf) in enumerate(clasificadores):
        print(f"Ejecutando: {name}")
        analyzer = MelodyAnalyzer(classifier=clf)
        result = analyzer.analyze_file(audio_path)
        
        # Usamos una versión modificada de plot_melody_contour o simplemente dibujamos en el eje
        # Para esta comparativa, usaremos el helper directamente sobre el eje
        from melody_analysis_v2.visualization import _draw_segment_overlays, _get_plot_step
        
        ax = axes[i]
        times = result.features.times
        pitch = result.features.pitch_midi
        step = _get_plot_step(len(times))
        
        ax.plot(times[::step], pitch[::step], color="tab:blue", alpha=0.8)
        ymax = float(np.nanmax(pitch)) if pitch.size else 60.0
        _draw_segment_overlays(ax, result.segments, ymax)
        
        ax.set_title(f"Clasificador: {name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Altura (MIDI)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Tiempo (s)")
    plt.tight_layout()
    
    save_path = output_dir / "comparativa_3_modelos.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparativa guardada en: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
