# MC-FSA

Herramienta experimental para segmentar y clasificar la estructura melódica de una grabación.
El flujo está inspirado en MSAF pero se centra en la melodía: extrae el contorno
(pitch y energía), detecta cambios estructurales y etiqueta cada frase con roles
musicales sencillos como "exposición", "pregunta" o "respuesta".

El detector de secciones combina dos pistas de cambio: una curva de novedad
derivada (saltos en pitch/energía) y una matriz de autosimilitud con núcleo
"checkerboard" para resaltar repeticiones y contrastes entre fragmentos.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Uso

Desde la línea de comandos:

```bash
python -m melody_analysis ruta/al/audio.wav \
    --output resultado.json \
    --melody-plot contorno.png \
    --sections-plot secciones.png
```

Los parámetros `--melody-plot` y `--sections-plot` guardan dos imágenes:
una con el contorno melódico extraído y otra con el espectrograma mel donde
se resaltan las secciones descritas en el JSON.

Si quieres experimentar con una copia independiente del pipeline sin tocar la
implementación original, hay un clon disponible bajo el nombre
`melody_analysis_v2` con los mismos puntos de entrada:

```bash
python -m melody_analysis_v2 ruta/al/audio.wav \
    --output resultado.json \
    --melody-plot contorno_v2.png \
    --sections-plot secciones_v2.png
```

En código:

```python
from melody_analysis import (
    MelodyClassifier,
    MelodyAnalyzer,
    plot_f0_no_segments,
    plot_f0_only,
    plot_melody_only,
    plot_melody_contour,
    plot_spectrogram_with_segments,
)
import librosa

analyzer = MelodyAnalyzer()
resultado = analyzer.analyze_file("ruta/al/audio.wav")
for segmento in resultado.segments:
    print(segmento.label, segmento.segment.start_time, segmento.segment.end_time)

# Generar visualizaciones directamente desde Python
fig1 = plot_melody_contour(resultado)
audio, sample_rate = librosa.load("ruta/al/audio.wav", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
# Si solo quieres el contorno melódico sin energía o f0:
fig_melodia = plot_melody_only(resultado)
# Si quieres únicamente la curva de f0 en Hz:
fig_f0 = plot_f0_only(resultado)
# Si prefieres f0 sin ninguna superposición de segmentos:
fig_f0_plano = plot_f0_no_segments(resultado)
# Señal normalizada + curvas de novedad (usa analyze_file/analyze_audio)
fig_novedad = plot_signal_and_novelty(resultado)
# Matriz de autosimilitud (pitch + energía)
fig_ssm = plot_self_similarity(resultado)
# Las gráficas incluyen el contorno en MIDI y la curva f0 (Hz) superpuesta cuando aplica.

# ¿Quieres renombrar las etiquetas (ej. "pregunta"→"Q" y "respuesta"→"A")?
# Solo cambia la línea donde se crea el analizador y pasa alias al clasificador;
# no hace falta tocar nada más y los colores/leyendas se conservan.
analyzer_custom = MelodyAnalyzer(
    classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"})
)
resultado_custom = analyzer_custom.analyze_file("ruta/al/audio.wav")
```

Si quieres usar directamente el clon con los colores mejorados (`melody_analysis_v2`)
desde un script como el que muestras (`from src.melody_analysis_v2 import ...`),
asegúrate primero de haber instalado el proyecto en editable (`pip install -e .`)
o de exportar `PYTHONPATH=src` antes de ejecutar el script. Luego importa sin el
prefijo `src` así:

```python
from melody_analysis_v2 import (
    MelodyAnalyzer,
    MelodyClassifier,
    plot_f0_no_segments,
    plot_f0_only,
    plot_melody_only,
    plot_melody_contour,
    plot_spectrogram_with_segments,
)
import librosa

analyzer = MelodyAnalyzer()
resultado = analyzer.analyze_file("1.mp3")
for segmento in resultado.segments:
    print(segmento.label, segmento.segment.start_time, segmento.segment.end_time)

# Para renombrar etiquetas en este mismo ejemplo, cambia la línea anterior por:
# analyzer = MelodyAnalyzer(
#     classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"})
# )

fig1 = plot_melody_contour(resultado)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
# Visualización minimalista solo del pitch en MIDI
fig_melodia = plot_melody_only(resultado)
# Visualización minimalista solo de f0 en Hz
fig_f0 = plot_f0_only(resultado)
# Visualización de f0 sin tramas de segmentos
fig_f0_plano = plot_f0_no_segments(resultado)
```

### Guía rápida para renombrar etiquetas a Q/A (o cualquier alias)

1. **Vía código (v1 o v2):** al crear el analizador, pasa `label_aliases` al
   clasificador. Solo necesitas modificar esa línea.

   ```python
   from melody_analysis import MelodyAnalyzer, MelodyClassifier  # o melody_analysis_v2

   analyzer = MelodyAnalyzer(
       classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"})
   )
   resultado = analyzer.analyze_file("ruta/al/audio.wav")
   ```

2. **Usando el ejemplo `examples/visualizar_melodia_v2.py`:** cambia la línea
   `analyzer = MelodyAnalyzer()` por la versión con alias anterior; no hay que
   tocar nada más en el script.

   - Ubicación exacta: el bloque de creación del analizador está al inicio del
     archivo. Sustituye la línea que crea el analizador por la línea comentada
     `# analyzer = MelodyAnalyzer(classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"}))`.
   - Si usas otro script, aplica la misma sustitución en la línea donde
     construyes `MelodyAnalyzer` o pasas explícitamente un `MelodyClassifier`.

### Checklist rápido para usar el visualizador en tu propio script

Si ya tienes un script parecido al ejemplo anterior y parece que "no hace
nada", verifica estos puntos:

1) Instala el proyecto en editable (`pip install -e .[dev]`) o exporta
   `PYTHONPATH=src` en la misma sesión antes de ejecutarlo. Así las
   importaciones `from melody_analysis_v2 import ...` funcionarán sin el
   prefijo `src.`
2) Llama a las funciones de visualización (`plot_melody_contour` y
   `plot_spectrogram_with_segments`) igual que en el snippet y guarda o
   muestra las figuras:

```python
fig1 = plot_melody_contour(resultado)
fig1.savefig("contorno.png", dpi=150)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, resultado)
fig2.savefig("secciones.png", dpi=150)
```

3) Si quieres ver las ventanas interactivas, exporta un backend con soporte
   gráfico, por ejemplo `MPLBACKEND=TkAgg`, o ejecuta el script en un entorno
   que ya tenga backend interactivo. Si Matplotlib queda en modo `Agg`, las
   figuras se guardarán en disco (como en el ejemplo anterior) y no se
   abrirán ventanas.

Siguiendo esos pasos, tu snippet estará usando el visualizador tal como en el
clon `melody_analysis_v2` con colores para cada clasificación.

Y de forma análoga puedes importar `MelodyAnalyzer` desde
`melody_analysis_v2` para modificarlo libremente sin afectar al módulo
original.

### Ejemplo paso a paso con el clon `melody_analysis_v2`

Si prefieres un script listo para ejecutar que explique línea a línea el
flujo completo y genere las dos imágenes de soporte, revisa
`examples/visualizar_melodia_v2.py`. El código contiene comentarios en
español que describen cada instrucción, imprime por consola los segmentos
detectados y guarda en `salidas_visualizacion/` tanto el contorno melódico
como los dos espectrogramas (manual y segmentado).

El script intenta usar un backend interactivo (TkAgg/QtAgg/MacOSX) si hay
soporte gráfico disponible, de modo que también puedas ver las ventanas con
`plt.show()`. Si Matplotlib continúa utilizando `Agg`, exporta la variable
de entorno `MPLBACKEND` con el backend de tu preferencia (por ejemplo,
`MPLBACKEND=TkAgg`) antes de ejecutar el script y asegúrate de tener las
dependencias correspondientes instaladas.

```bash
python examples/visualizar_melodia_v2.py
```

Solo necesitas sustituir la ruta `1.mp3` que aparece en el script por tu
archivo de audio antes de ejecutarlo.

### Visualizaciones por etapa (v2)

Además de las vistas de contorno y espectrograma, el clon `melody_analysis_v2`
expone helpers separados para ver cada fase del pipeline en imágenes
independientes:

- `plot_self_similarity(resultado)`: matriz de autosimilitud.
- `plot_boundary_detection(resultado)`: curvas de novedad (derivadas, SSM y
  combinada) usadas para detectar fronteras.
- `plot_segment_extraction(resultado)`: franjas de segmentos ya detectados
  sobre una línea de tiempo.
- `plot_descriptor_summary(resultado)`: barras por descriptor por segmento.

Ejemplo reducido:

```python
from melody_analysis_v2 import (
    MelodyAnalyzer,
    plot_self_similarity,
    plot_boundary_detection,
    plot_segment_extraction,
    plot_descriptor_summary,
)

resultado = MelodyAnalyzer().analyze_file("1.mp3")
plot_self_similarity(resultado).savefig("autosimilitud.png")
plot_boundary_detection(resultado).savefig("boundary_detection.png")
plot_segment_extraction(resultado).savefig("segment_extraction.png")
plot_descriptor_summary(resultado).savefig("descriptor_summary.png")
```

## Diagrama de flujo del análisis

El pipeline de `MelodyAnalyzer` (y su clon `melody_analysis_v2`) sigue estos
pasos de izquierda a derecha:

```mermaid
flowchart LR
    A[Audio de entrada<br/>WAV/MP3] --> B[Extracción STFT<br/>+ espectrograma mel]
    B --> C[Estimación de contorno<br/>f0 en MIDI y Hz]
    C --> D[Suavizado y normalización<br/>pitch/energía]
    D --> E[Curva de novedad
             basada en derivadas]
    D --> F[Matriz de autosimilitud
             con núcleo checkerboard]
    E --> G[Fusión de pistas de cambio<br/>novelty + autosimilitud]
    F --> G
    G --> H[Detección de límites<br/>de segmentos]
    H --> I[Cálculo de descriptores<br/>por segmento
            (pendiente, rango,
             tensión paramétrica)]
    I --> J[Clasificador heurístico
            (exposición, pregunta,
             respuesta, etc.)]
    J --> K[Visualización
            contorno + f0 + colores
            por etiqueta]
    J --> L[Export JSON
            con etiquetas
            y descriptores]
```

- **Autosimilitud**: compara ventanas del contorno para resaltar repeticiones o
  contrastes; se filtra con un núcleo tipo checkerboard para obtener una curva
  de novedad adicional.
- **Fusión de pistas**: la curva de novedad derivada (saltos en pitch/energía)
  se combina con la curva proveniente de autosimilitud; los picos resultantes
  definen los posibles cortes de frase.
- **Clasificación**: cada segmento recibe descriptores de contorno, rango,
  energía y tensión; un clasificador por reglas asigna roles musicales
  sencillos (exposición, pregunta, respuesta, transición, etc.).

## Pruebas

```bash
pytest
```
