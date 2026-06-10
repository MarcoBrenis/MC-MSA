# MC-MSA

An experimental tool to segment and classify the melodic structure of an audio recording.
The workflow is inspired by MSAF but focuses on melody: it extracts the contour
(pitch and energy), detects structural changes, and labels each phrase with simple
musical roles such as "exposition", "question", or "answer".

The section detector combines two change cues: a derived novelty curve
(pitch/energy jumps) and a self-similarity matrix with a checkerboard kernel
to highlight repetitions and contrasts between fragments.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Usage

From the command line:

```bash
python -m melody_analysis path/to/audio.wav \
    --output result.json \
    --melody-plot contour.png \
    --sections-plot sections.png
```

The `--melody-plot` and `--sections-plot` parameters save two images:
one with the extracted melodic contour and another with the mel spectrogram where
the sections described in the JSON are highlighted.

If you want to experiment with an independent copy of the pipeline without modifying the
original implementation, a clone is available under the name
`melody_analysis_v2` with the same entry points:

```bash
python -m melody_analysis_v2 path/to/audio.wav \
    --output result.json \
    --melody-plot contour_v2.png \
    --sections-plot sections_v2.png
```

In code:

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
result = analyzer.analyze_file("path/to/audio.wav")
for segment in result.segments:
    print(segment.label, segment.segment.start_time, segment.segment.end_time)

# Generate visualizations directly from Python
fig1 = plot_melody_contour(result)
audio, sample_rate = librosa.load("path/to/audio.wav", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, result)
# If you only want the melodic contour without energy or f0:
fig_melody = plot_melody_only(result)
# If you only want the f0 curve in Hz:
fig_f0 = plot_f0_only(result)
# If you prefer f0 without any segment overlay:
fig_f0_flat = plot_f0_no_segments(result)
# Normalized signal + novelty curves (uses analyze_file/analyze_audio)
fig_novelty = plot_signal_and_novelty(result)
# Self-similarity matrix (pitch + energy)
fig_ssm = plot_self_similarity(result)
# Plots include the contour in MIDI and the f0 curve (Hz) overlaid when applicable.

# Do you want to rename the labels (e.g. "pregunta" -> "Q" and "respuesta" -> "A")?
# Just change the line where the analyzer is created and pass aliases to the classifier;
# no need to modify anything else, and colors/legends are preserved.
analyzer_custom = MelodyAnalyzer(
    classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"})
)
result_custom = analyzer_custom.analyze_file("path/to/audio.wav")
```

If you want to directly use the clone with improved colors (`melody_analysis_v2`)
from a script like the one shown (`from src.melody_analysis_v2 import ...`),
make sure first to have installed the project in editable mode (`pip install -e .`)
or export `PYTHONPATH=src` before running the script. Then import without the
`src` prefix like this:

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
result = analyzer.analyze_file("1.mp3")
for segment in result.segments:
    print(segment.label, segment.segment.start_time, segment.segment.end_time)

# To rename labels in this same example, change the line above to:
# analyzer = MelodyAnalyzer(
#     classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"})
# )

fig1 = plot_melody_contour(result)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, result)
# Minimalist visualization of pitch in MIDI only
fig_melody = plot_melody_only(result)
# Minimalist visualization of f0 in Hz only
fig_f0 = plot_f0_only(result)
# Visualization of f0 without segment overlays
fig_f0_flat = plot_f0_no_segments(result)
```

### Quick Guide to Rename Labels to Q/A (or any alias)

1. **Via code (v1 or v2):** when creating the analyzer, pass `label_aliases` to the
   classifier. You only need to modify that line.

   ```python
   from melody_analysis import MelodyAnalyzer, MelodyClassifier  # or melody_analysis_v2

   analyzer = MelodyAnalyzer(
       classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"})
   )
   result = analyzer.analyze_file("path/to/audio.wav")
   ```

2. **Using the example `examples/visualizar_melodia_v2.py`:** change the line
   `analyzer = MelodyAnalyzer()` to the aliased version above; no other changes
   are needed in the script.

   - Exact location: the block for creating the analyzer is at the beginning of the
     file. Replace the line that creates the analyzer with the commented version
     `# analyzer = MelodyAnalyzer(classifier=MelodyClassifier(label_aliases={"pregunta": "Q", "respuesta": "A"}))`.
   - If using another script, apply the same substitution on the line where you
     construct `MelodyAnalyzer` or explicitly pass a `MelodyClassifier`.

### Quick Checklist to Use the Visualizer in Your Own Script

If you already have a script similar to the previous example and it seems to "do
nothing", check these points:

1) Install the project in editable mode (`pip install -e .[dev]`) or export
   `PYTHONPATH=src` in the same session before running it. This ensures the
   imports `from melody_analysis_v2 import ...` work without the `src.` prefix.
2) Call the visualization functions (`plot_melody_contour` and
   `plot_spectrogram_with_segments`) just like in the snippet and save or
   show the figures:

```python
fig1 = plot_melody_contour(result)
fig1.savefig("contour.png", dpi=150)
audio, sample_rate = librosa.load("1.mp3", sr=22050)
fig2 = plot_spectrogram_with_segments(audio, sample_rate, result)
fig2.savefig("sections.png", dpi=150)
```

3) If you want to see interactive windows, export a backend with graphical support,
   for example `MPLBACKEND=TkAgg`, or run the script in an environment that
   already has an interactive backend. If Matplotlib remains in `Agg` mode, the
   figures will be saved to disk (as in the previous example) and no windows will open.

Following these steps, your snippet will use the visualizer just like in the clone
`melody_analysis_v2` with colors for each classification.

Analogously, you can import `MelodyAnalyzer` from `melody_analysis_v2` to
modify it freely without affecting the original module.

### Step-by-Step Example with the `melody_analysis_v2` Clone

If you prefer a ready-to-run script that explains the entire flow line by line and
generates both support images, check `examples/visualizar_melodia_v2.py`. The code
contains comments in Spanish describing each instruction, prints the detected
segments to the console, and saves both the melodic contour and the two spectrograms
(manual and segmented) in `salidas_visualizacion/`.

The script attempts to use an interactive backend (TkAgg/QtAgg/MacOSX) if graphical
support is available, so you can also see the windows with `plt.show()`. If Matplotlib
continues using `Agg`, export the `MPLBACKEND` environment variable with your
preferred backend (for example, `MPLBACKEND=TkAgg`) before running the script and
make sure you have the corresponding dependencies installed.

```bash
python examples/visualizar_melodia_v2.py
```

You only need to replace the path `1.mp3` in the script with your audio file before running it.

### Stage-by-Stage Visualizations (v2)

In addition to the contour and spectrogram views, the `melody_analysis_v2` clone
exposes separate helpers to view each phase of the pipeline in independent images:

- `plot_self_similarity(result)`: self-similarity matrix.
- `plot_boundary_detection(result)`: novelty curves (derived, SSM, and combined)
  used to detect boundaries.
- `plot_segment_extraction(result)`: timelines of already detected segments.
- `plot_descriptor_summary(result)`: bars per descriptor per segment.

Reduced example:

```python
from melody_analysis_v2 import (
    MelodyAnalyzer,
    plot_self_similarity,
    plot_boundary_detection,
    plot_segment_extraction,
    plot_descriptor_summary,
)

result = MelodyAnalyzer().analyze_file("1.mp3")
plot_self_similarity(result).savefig("self_similarity.png")
plot_boundary_detection(result).savefig("boundary_detection.png")
plot_segment_extraction(result).savefig("segment_extraction.png")
plot_descriptor_summary(result).savefig("descriptor_summary.png")
```

## Analysis Flowchart

The `MelodyAnalyzer` pipeline (and its clone `melody_analysis_v2`) follows these
steps from left to right:

```mermaid
flowchart LR
    A[Input Audio<br/>WAV/MP3] --> B[STFT Extraction<br/>+ Mel Spectrogram]
    B --> C[Contour Estimation<br/>f0 in MIDI and Hz]
    C --> D[Smoothing and Normalization<br/>pitch/energy]
    D --> E[Novelty Curve
             based on derivatives]
    D --> F[Self-Similarity Matrix
             with checkerboard kernel]
    E --> G[Fusion of change cues<br/>novelty + self-similarity]
    F --> G
    G --> H[Boundary Detection<br/>of segments]
    H --> I[Descriptor Calculation<br/>per segment
            (slope, range,
             parametric tension)]
    I --> J[Heuristic Classifier
            (exposition, question,
             answer, etc.)]
    J --> K[Visualization
            contour + f0 + colors
            by label]
    J --> L[JSON Export
            with labels
            and descriptors]
```

- **Self-Similarity**: compares windows of the contour to highlight repetitions or
  contrasts; it is filtered with a checkerboard kernel to obtain an additional novelty curve.
- **Cue Fusion**: the derived novelty curve (pitch/energy jumps) is combined with the curve
  coming from self-similarity; the resulting peaks define the possible phrase boundaries.
- **Classification**: each segment receives contour, range, energy, and tension descriptors;
  a rule-based classifier assigns simple musical roles (exposition, question, answer, transition, etc.).

## Tests

```bash
pytest
```
