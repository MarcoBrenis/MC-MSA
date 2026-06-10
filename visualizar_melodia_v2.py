"""Ejemplo comentado para analizar una melodía y generar visualizaciones."""

# --- Importación de librerías ---
from pathlib import Path  # Para manejar rutas de archivos al guardar imágenes.
import sys

# Se importan las herramientas necesarias para el análisis y la visualización.
import soundfile as sf  # Para leer archivos de audio.
import librosa  # Para rutinas de análisis de audio complementarias.
import librosa.display  # Para representar el espectrograma en un eje tiempo-frecuencia.
import matplotlib  # Para consultar qué backend se está usando.
import matplotlib.pyplot as plt  # Para crear y mostrar gráficos.
import numpy as np  # Para operaciones numéricas, especialmente con arrays.

# Se importa la clase principal y los auxiliares de visualización del clon v2.
from src.melody_analysis_v2 import (
    MelodyAnalyzer,  # Encapsula la extracción, segmentación y clasificación.
    MelodyClassifierPaper,  # Nuevo clasificador para el paper.
    plot_melody_contour,  # Función para graficar el contorno melódico.
    plot_melody_only,  # Función para graficar solo el contorno melódico sin segmentos ni energía.
    plot_energy_only,  # Función para graficar solo la energía.
    plot_melody_and_energy,  # Función para graficar contorno y energía.
    plot_spectrogram_with_segments,  # Función para graficar el espectrograma con secciones.
    synthesize_melody,  # Función para sintetizar la melodía extraída.
)
from src.melody_analysis_v2.visualization import (
    plot_self_similarity,
    plot_boundary_detection,
)


def main() -> None:
    """Ejecuta el análisis sobre un archivo y muestra los resultados."""

    # --- Análisis de la Melodía ---
    # Se define la ruta al archivo de audio que se quiere analizar.
    # Usamos una ruta relativa al script para mayor robustez.
    script_dir = Path(__file__).parent.absolute()
    audio_path = script_dir / "1.mp3"
    
    if not audio_path.exists():
        print(f"Error: No se encuentra el archivo de audio en {audio_path}")
        return

    # --- Configuración del Análisis ---
    print("Elija el método de extracción de melodía:")
    print("1. pyin (Probabilistic YIN - Rápido)")
    print("2. crepe (Deep Learning - Preciso)")
    print("3. ensemble (Combinación de ambos - Robusto)")
    print("4. melodia (Essentia - Estándar clásico para polifonía)")
    print("5. demucs_crepe (Separación de voz + CREPE)")
    print("6. bs_roformer_rmvpe (BS-RoFormer + RMVPE)")
    print("7. rmvpe (Solo RMVPE)")
    opcion = input("Opción [1-7] (default 1): ").strip()
    
    metodos = {
        "1": "pyin", 
        "2": "crepe", 
        "3": "ensemble",
        "4": "melodia",
        "5": "demucs_crepe",
        "6": "bs_roformer_rmvpe",
        "7": "rmvpe"
    }
    metodo_elegido = metodos.get(opcion, "pyin")
    
    # --- Configuración del Clasificador ---
    print("\nElija el clasificador:")
    print("1. Estándar (Caplin Completo: Antecedente, Consecuente, Presentación, etc.)")
    print("2. Paper CLEI (Estricto A, C, X)")
    op_clf = input("Opción [1/2] (default 1): ").strip()
    
    if op_clf == "2":
        classifier = MelodyClassifierPaper()
        print("Usando clasificador del Paper (A/C/X).")
    else:
        classifier = None # Usará el default MelodyClassifier
        print("Usando clasificador Estándar.")

    # Se crea una instancia del analizador de melodías con el método y clasificador elegidos.
    analyzer = MelodyAnalyzer(extraction_method=metodo_elegido, classifier=classifier)
    # Se llama al método para analizar el archivo, que devuelve un objeto con los resultados.
    try:
        resultado = analyzer.analyze_file(str(audio_path))
    except ImportError as exc:
        print(f"\nNo se pudo ejecutar el método '{metodo_elegido}': {exc}")
        if metodo_elegido == "crepe":
            print(
                "\nSugerencias:\n"
                "- Usa la opción 1 ('pyin') para seguir trabajando sin TensorFlow.\n"
                "- Si necesitas CREPE, crea un entorno virtual con una versión de Python "
                "compatible con TensorFlow y vuelve a instalar dependencias."
            )
        elif metodo_elegido == "ensemble":
            print(
                "\nSugerencia:\n"
                "- 'ensemble' también depende de CREPE. Si TensorFlow no está disponible, "
                "usa la opción 1 ('pyin')."
            )
        elif metodo_elegido == "demucs_crepe":
            print(
                "\nSugerencia:\n"
                "- 'demucs_crepe' también termina usando CREPE. Si TensorFlow no está "
                "disponible, prueba primero con 'pyin' o 'melodia'."
            )
        sys.exit(1)

    # Se imprime en la consola un resumen de los segmentos detectados.
    print("Segmentos formalmente clasificados (Reglas de Caplin):")
    # Se recorre cada segmento en los resultados para mostrar sus datos principales.
    for segmento in resultado.segments:
        # Extraemos el score de similitud SSM si existe, o 0.0 si es el primer segmento
        sim_score = segmento.descriptor.get("ssm_similarity_with_previous", 0.0)
        # Se imprime la etiqueta, el score de similitud y los tiempos
        print(f"{segmento.label:>20} (SSM Sim: {sim_score:4.2f}) | {segmento.segment.start_time:7.3f} → {segmento.segment.end_time:7.3f} s")

    # Se consulta el backend activo de Matplotlib para decidir si se mostrarán ventanas.
    backend = matplotlib.get_backend()
    backend_lower = backend.lower()
    interactive_backend = not backend_lower.endswith("agg")

    if interactive_backend:
        print(f"El backend de Matplotlib es '{backend}', se mostrarán las figuras al final.")
    else:
        print(
            "Backend no interactivo (Agg u otro similar): las figuras se guardarán en disco.\n"
            "Para abrir ventanas interactivas puedes definir MPLBACKEND=TkAgg (u otro backend)\n"
            "antes de ejecutar el script, siempre que tengas las dependencias instaladas."
        )

    # --- Visualización del contorno melódico ---
    # Se asegura un directorio de salida para guardar las figuras generadas.
    # Ahora organizamos por método de extracción.
    output_base_dir = script_dir / "salidas_visualizacion"
    output_dir = output_base_dir / metodo_elegido
    output_dir.mkdir(parents=True, exist_ok=True)

    # Se obtiene una figura con el contorno melódico usando el helper incluido en el paquete.
    contour_fig = plot_melody_contour(resultado)
    # Se define la ruta donde se guardará la imagen del contorno melódico.
    contour_path = output_dir / "contorno_melodico.png"
    # Se guarda la figura en disco con buena resolución.
    contour_fig.savefig(contour_path, dpi=150, bbox_inches="tight")
    # Si no se dispone de backend interactivo, se cierra la figura para liberar recursos.
    if not interactive_backend:
        plt.close(contour_fig)
    # Se informa en consola dónde quedó almacenada la imagen.
    print(f"Figura de contorno guardada en: {contour_path}")

    # Se obtiene una figura de solo el contorno melódico sin segmentos ni energía.
    melody_only_fig = plot_melody_only(resultado, show_segments=False)
    # Se define la ruta donde se guardará la imagen.
    melody_only_path = output_dir / "contorno_melodico_solo.png"
    melody_only_fig.savefig(melody_only_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(melody_only_fig)
    print(f"Figura de solo contorno guardada en: {melody_only_path}")

    # Se obtiene una figura de solo la energía.
    energy_only_fig = plot_energy_only(resultado)
    energy_only_path = output_dir / "energia_solo.png"
    energy_only_fig.savefig(energy_only_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(energy_only_fig)
    print(f"Figura de solo energía guardada en: {energy_only_path}")

    # Se obtiene una figura de contorno melódico y energía.
    melody_energy_fig = plot_melody_and_energy(resultado)
    melody_energy_path = output_dir / "contorno_y_energia.png"
    melody_energy_fig.savefig(melody_energy_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(melody_energy_fig)
    print(f"Figura de contorno y energía guardada en: {melody_energy_path}")

    # --- Novedad y Matrices SSM ---
    try:
        ssm_fig = plot_self_similarity(resultado)
        ssm_path = output_dir / "matriz_autosimilitud.png"
        ssm_fig.savefig(ssm_path, dpi=150, bbox_inches="tight")
        if not interactive_backend:
            plt.close(ssm_fig)
        print(f"Matriz de autosimilitud guardada en: {ssm_path}")
    except Exception as e:
        print(f"No se pudo graficar la matriz SSM: {e}")

    try:
        bound_fig = plot_boundary_detection(resultado)
        bound_path = output_dir / "deteccion_fronteras.png"
        bound_fig.savefig(bound_path, dpi=150, bbox_inches="tight")
        if not interactive_backend:
            plt.close(bound_fig)
        print(f"Curvas de detección de fronteras guardadas en: {bound_path}")
    except Exception as e:
        print(f"No se pudo graficar la detección de fronteras: {e}")

    # --- Visualización manual del Mel-espectrograma ---
    # Se carga el archivo de audio con soundfile para acceder a la señal (y) y la frecuencia de muestreo (sr).
    y, sr = sf.read(str(audio_path))
    # Si el audio es estéreo (más de un canal), se convierte a mono promediando los canales.
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Se calcula el Mel-espectrograma; representa la energía por bandas perceptuales a lo largo del tiempo.
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # Se convierte la potencia a decibelios para resaltar detalles finos.
    Sdb = librosa.power_to_db(S, ref=np.max)

    # Se crea un gráfico del espectrograma en Matplotlib y se obtiene la figura resultante.
    manual_fig, ax = plt.subplots(figsize=(14, 4))
    # Se muestra el espectrograma en el gráfico, con ejes de tiempo y frecuencia en escala Mel.
    librosa.display.specshow(Sdb, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    # Se añade un título descriptivo al gráfico.
    ax.set_title("Mel-espectrograma (manual)")
    # Se ajusta el diseño para evitar solapamiento de elementos.
    manual_fig.tight_layout()
    # Se guarda la figura manual para contrastarla posteriormente con la versión anotada.
    manual_path = output_dir / "mel_espectrograma_manual.png"
    manual_fig.savefig(manual_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(manual_fig)
    # Se informa la ruta de la figura manual.
    print(f"Mel-espectrograma manual guardado en: {manual_path.resolve()}")

    # --- Visualización automatizada con secciones ---
    # Se genera una figura que reutiliza los segmentos detectados para resaltar cada bloque musical.
    sections_fig = plot_spectrogram_with_segments(y, sr, resultado)
    # Se guarda la figura con las secciones y etiquetas incrustadas.
    sections_path = output_dir / "mel_espectrograma_segmentado.png"
    sections_fig.savefig(sections_path, dpi=150, bbox_inches="tight")
    if not interactive_backend:
        plt.close(sections_fig)
    # Se muestra la ruta donde quedó guardado el espectrograma con secciones.
    print(f"Mel-espectrogramas segmentado guardado en: {sections_path.resolve()}")

    # --- Exportación de Audio de la Melodía ---
    try:
        print("Sintetizando audio de la melodía...")
        # Sintetizamos la señal de audio a partir de las características extraídas.
        audio_melodia = synthesize_melody(
            resultado.features.times,
            resultado.features.pitch_midi,
            resultado.features.confidence,
            resultado.features.energy,
            sample_rate=sr
        )
        # Definimos la ruta de salida para el archivo de audio.
        audio_output_path = output_dir / "melodia_extraida.wav"
        # Guardamos el archivo de audio usando soundfile.
        sf.write(str(audio_output_path), audio_melodia, sr)
        print(f"Audio de la melodía guardado en: {audio_output_path.resolve()}")
    except Exception as e:
        print(f"Error al exportar el audio de la melodía: {e}")

    # Si hay backend interactivo se realiza un plt.show() final para abrir las ventanas.
    if interactive_backend:
        print("Abriendo ventanas de Matplotlib...")
        plt.show()


if __name__ == "__main__":
    main()
