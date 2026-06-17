# MC-MSA v2.5: Melody Extraction & Evaluation Guide

This user guide provides details on using the evaluation suite of the **MC-MSA v2.5** framework. The framework evaluates melody extraction methods (F0 estimation, vocal separation hybrids, and symbolic estimators) on Cover Song Identification (CSI) tasks using Information Retrieval (IR) metrics.

---

## 1. Directory & Dataset Structure

The pipeline dynamically detects dataset folders in the workspace that prefix with `dataset_` (e.g., `dataset_OA`, `dataset_Acad`, `dataset_JJCC`).

Each dataset folder must contain two subdirectories:
* **`originales/`**: The reference/original audio tracks.
* **`covers/`**: The target cover versions.

### Matching Modes
When running the evaluation, you can select how tracks are paired between `originales/` and `covers/`:
1. **`id` (Numeric ID Prefix - Default)**: Pairs tracks starting with the same number (e.g., `04 - Song A.mp3` matches `04 - Cover.wav`).
2. **`stem` (Exact Name)**: Matches normalized file names.
3. **`fuzzy` (Smart/Fuzzy Match)**: Utilizes word-overlap calculations for complex naming schemes or classical metadata.

---

## 2. Core Evaluation Scripts

### A. Full Melody Extraction Evaluation (`run_mc_msa.py`)
This is the main runner. It iterates through the dataset, extracts melodic contours using the selected method(s), performs segmentation/classification, and calculates retrieval metrics.

* **Usage**:
  ```bash
  python run_mc_msa.py
  ```
* **Arguments**:
  * `--method`: Extraction method to evaluate (e.g., `pyin`, `crepe`, `demucs_crepe`, `bs_roformer_rmvpe`, `all`, etc.).
  * `--dataset_dir`: Name or absolute path of the dataset folder (e.g., `dataset_OA`).
  * `--match_mode`: Pairing mode (`id`, `stem`, or `fuzzy`).
  * `--cache_dir`: Directory for caching segmentations (`default: cache`).
  * `--clear_cache`: Clear cached data for the selected method before starting.
  * `--dtw_all_pairs`: Calculate Dynamic Time Warping (DTW) for all pairs (Warning: computationally expensive).

---

### B. Hyperparameter Optimization (`run_mc_msa_optuna.py`)
Optimizes the parameters of `MelodyClassifierPaper` (voicing threshold, slope epsilon, energy tau) using **Optuna** to maximize the Mean Reciprocal Rank (MRR) or Longest Common Subsequence (LCS) similarity.

* **Usage**:
  ```bash
  python run_mc_msa_optuna.py --dataset_dir dataset_OA --method pyin
  ```
* **Notes**:
  * Leverages cached features to run thousands of parameter trials in-memory in seconds.
  * Outputs the optimized parameters to `resultados_mc_msa_optuna/mc_msa_summary.csv`.

---

### C. Fast Recalculation of Metrics (`run_metrics_only.py`)
Recalculates all IR metrics and regenerates report tables **instantaneously** from cached analysis files without reprocessing the audio signals. Use this after changing metric definitions or to review summary tables.

* **Usage**:
  ```bash
  python run_metrics_only.py
  ```
* **Arguments**:
  * `--dataset_dir`: Target dataset folder.
  * `--method`: Recalculate a specific method or `all` available methods in the cache.
  * `--optuna`: Recalculate metrics using the Optuna-optimized results.

---

### D. Tiny Evaluation Runner (`run_tiny_mc_msa.py`)
A lightweight script designed for testing or debugging. It executes the analysis pipeline on a subset of two song pairs (IDs 02 and 04) and generates full comparative and qualitative figures.

* **Usage**:
  ```bash
  python run_tiny_mc_msa.py --method pyin
  ```
* **Outputs**:
  * `salidas_tiny_mc_msa/<method>/detailed_report.txt`: English performance summary.
  * `fig_qualitative_bands.pdf` & `fig_qualitative_contour.pdf`: Segment comparisons.
  * Spectrograms, novelty curves, and self-similarity matrices (SSM) for the best match.

---

### E. Single Track Analyzer (`analyze_single_track.py`)
Analyzes a single audio file, prints a table of detected formal segments, and exports full visualizations and a synthesized melody WAV file.

* **Usage**:
  ```bash
  python analyze_single_track.py path/to/song.mp3 --method pyin --classifier standard
  ```
* **Arguments**:
  * `audio_path`: Path to the audio file (optional, prompts if omitted).
  * `--method`: Pitch extraction method (e.g., `pyin`, `crepe`, `bs_roformer_rmvpe`, etc.).
  * `--classifier`: Classifier rules (`standard` or `paper`).
  * `--output_dir`: Output directory for exports (default: `salidas_single_track`).
* **Outputs**:
  * `<song>_contour.png`: Melodic contour with segments and energy.
  * `<song>_spectrogram.png`: Segment-annotated Mel-spectrogram.
  * `<song>_ssm.png`: Self-similarity matrix.
  * `<song>_synthesized.wav`: Sine-wave synthesized audio of the extracted melody contour.

---

### F. Real-Audio Thesis Figure Generator (`generate_thesis_figures.py`)
> [!NOTE]
> **Local Research Utility:** This script is preserved locally for thesis visualization research but is excluded from Git version control.
>
> Generates academic-quality visualizations of the Self-Similarity Matrix (SSM) based on a real audio file, showing detected boundaries, homogeneous blocks, and melodic repetitions.

* **Usage**:
  ```bash
  python generate_thesis_figures.py path/to/song.mp3 --method pyin
  ```
* **Outputs**:
  * `ssm_boundaries.png`: The SSM showing boundaries along the diagonal.
  * `ssm_homogeneous.png`: The SSM highlighting a cohesive diagonal block.
  * `ssm_repetitions.png`: The SSM highlighting symmetric off-diagonal repetitions.

---

### G. Conceptual Thesis Figure Generator (`generate_thesis_plots.py`)
> [!NOTE]
> **Local Research Utility:** This script is preserved locally for thesis visualization research but is excluded from Git version control.
>
> Generates clean, noise-free conceptual diagrams of an SSM representing a classic structure ($A$-$B$-$A'$-$B'$). Ideal for explaining theoretical concepts in the text.

* **Usage**:
  ```bash
  python generate_thesis_plots.py
  ```
* **Outputs**:
  * `fig_homogeneity.png`: Two-panel visualization showing a full SSM and a zoomed-in homogeneous block on the diagonal (Homogeneity Principle).
  * `fig_repetition.png`: Multi-panel visualization comparing the diagonal blocks ($A_1$ and $A_2$) and their off-diagonal intersection ($A_1 \times A_2$), demonstrating parallel similarity patterns (Repetition Principle).

---

## 3. Caching & Memory Management

To support large datasets (70+ pairs) on consumer hardware without Out-Of-Memory (OOM) crashes:
* **Audio Delegation**: Computational tasks are processed via isolated subprocesses (`analyze_single.py`) to prevent RAM leaks from TensorFlow/PyTorch runtimes.
* **Feature Caching**: Melody sequences and pitch arrays are stored in `cache/<method>/` as lightweight `.json` files.
* **Separation Cache**: Extracted vocals from Demucs or BS-Roformer are cached as `.npy` arrays inside `src/melody_analysis_v2/.vocal_cache/` to avoid repeated source separation.

---

## 4. Evaluation Metrics

The evaluations calculate the following Information Retrieval metrics:
* **LCS (%)**: Longest Common Subsequence normalized similarity.
* **MR**: Mean Rank of the correct match (lower is better).
* **MDR**: Median Rank of the correct match (lower is better).
* **MRR (%)**: Mean Reciprocal Rank (primary metric, higher is better).
* **MAP (%)**: Mean Average Precision.
* **Top-5 / Top-10 (%)**: Percentage of targets found in the top 5 or 10 ranks.
* **DTW**: Average Dynamic Time Warping distance between pitch contours.

## 5. Environment Setup & Installation

Follow these steps to set up a clean Python environment and install all dependencies:

### A. System Prerequisites
The audio loading engines (`librosa`, `soundfile`) require **FFmpeg** to decode formats like `.mp3` and `.wav`.
* **macOS**: `brew install ffmpeg`
* **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
* **Windows**: Download and add the FFmpeg binary path to your system's `PATH`.

### B. Create and Activate the Virtual Environment
Create a virtual environment named `.venv` using Python 3.10 or higher:
```bash
# Create the environment
python3 -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate

# Activate on Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### C. Install Dependencies
You can choose the installation configuration that fits your usage:

* **Basic installation** (Only basic DSP extractors like `pyin` or `yin`):
  ```bash
  pip install -e .
  ```
* **Developer/Test tools installation** (Includes testing libraries like `pytest`):
  ```bash
  pip install -e .[dev]
  ```
* **Advanced/Deep Learning installation** (Includes heavy neural extractors like BS-Roformer, Demucs, basic-pitch, and PyTorch):
  ```bash
  pip install -e .[advanced]
  ```
* **Full installation** (Recommended for the complete evaluation suite, including hyperparameter tuning):
  ```bash
  pip install -e .[advanced,optuna]
  ```

---

### Supported Extraction Methods
* **Basic Pitch**: Spotify's lightweight polyphonic transcription model.
* **BS-Roformer + RMVPE**: High-fidelity vocal extraction combined with robust CNN-based pitch estimation (Apple Silicon MPS / Nvidia CUDA accelerated).
* **Demucs + CREPE**: Traditional hybrid source separation and deep F0 tracking.
* **Melodia**: Melodic salience algorithm (requires `essentia` package).
* **pYIN / YIN**: Classic DSP pitch tracking.
