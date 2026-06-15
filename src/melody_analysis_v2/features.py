"""Feature extraction utilities for melody analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import gc

import numpy as np

try:
    import librosa
    from scipy.interpolate import interp1d
except Exception:  # pragma: no cover - handled gracefully
    librosa = None  # type: ignore
    interp1d = None

try:
    import crepe
except ImportError:
    crepe = None

try:
    import essentia.standard as es
except ImportError:
    es = None

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
except ImportError:
    torch = None
    get_model = None
    apply_model = None

try:
    import tensorflow_hub as hub
except ImportError:
    hub = None


# Global model cache to prevent OOM
_MODEL_CACHE = {}

@dataclass
class MelodyFeatures:
    """Container for features derived from a melody contour."""

    times: np.ndarray
    """Time stamps (seconds) for each frame."""

    pitch_midi: np.ndarray
    """Estimated fundamental frequency expressed in MIDI note numbers."""

    confidence: np.ndarray
    """Confidence of the pitch estimate for each frame in the range [0, 1]."""

    energy: np.ndarray
    """Normalized energy for each frame."""

    @property
    def duration(self) -> float:
        """Return the duration of the feature sequence."""

        if self.times.size == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Interpolate NaN values using linear interpolation."""

    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.zeros_like(values)

    nans = np.isnan(values)
    if not np.any(nans):
        return values

    indices = np.arange(values.size)
    values[nans] = np.interp(indices[nans], indices[~nans], values[~nans])
    return values


def _extract_pyin(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    frame_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for pYIN extraction."""
    pitch_hz, _, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    pitch_midi = librosa.hz_to_midi(pitch_hz)
    confidence = np.where(np.isnan(voiced_prob), 0.0, voiced_prob)
    return pitch_midi, confidence

def _extract_yin(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    frame_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for YIN extraction."""
    pitch_hz = librosa.yin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    pitch_midi = librosa.hz_to_midi(pitch_hz)
    # YIN doesn't provide confidence, use 1.0 for all frames as a baseline
    confidence = np.ones_like(pitch_hz)
    return pitch_midi, confidence


def _extract_crepe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for CREPE extraction."""
    if crepe is None:
        raise ImportError("crepe is required for 'crepe' method but not installed.")

    # crepe expects step_size in milliseconds
    step_size = int(1000 * hop_length / sample_rate)

    try:
        # crepe.predict returns time, frequency, confidence, activation
        _, frequency, confidence, _ = crepe.predict(
            audio, sample_rate, step_size=step_size, model_capacity='tiny', verbose=0
        )
    except ModuleNotFoundError as exc:
        missing_module = exc.name or str(exc)
        if "tensorflow" in missing_module.lower():
            raise ImportError(
                "El método 'crepe' requiere TensorFlow además del paquete 'crepe'. "
                "En este entorno no está disponible. Si usas macOS, TensorFlow suele "
                "tener compatibilidad limitada con versiones recientes de Python "
                "(por ejemplo Python 3.13). Usa 'pyin' como alternativa inmediata "
                "o crea un entorno con una versión de Python compatible con TensorFlow."
            ) from exc
        raise

    # Convert frequency (Hz) to MIDI
    pitch_midi = librosa.hz_to_midi(frequency)
    return pitch_midi, confidence


def _extract_melodia(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Melodia (Essentia) extraction."""
    if es is None:
        raise ImportError("essentia is required for 'melodia' method but not installed.")

    # PredominantPitchMelodia expects float32
    audio_es = audio.astype(np.float32)

    # Note: hopSize in Melodia is in samples.
    # We want it to match our hop_length for alignment.
    melodia = es.PredominantPitchMelodia(
        sampleRate=sample_rate,
        hopSize=hop_length,
    )

    pitch_hz, confidence = melodia(audio_es)

    # Convert Hz to MIDI
    # Pitch is 0 for unvoiced/silence
    pitch_midi = np.zeros_like(pitch_hz)
    voiced = pitch_hz > 0
    pitch_midi[voiced] = librosa.hz_to_midi(pitch_hz[voiced])
    pitch_midi[~voiced] = np.nan

    return pitch_midi, confidence


def _extract_tachibana(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Tachibana's HPSS-based melody extraction (HPSS + Melodia)."""
    if librosa is None:
        raise ImportError("librosa is required for HPSS separation in 'tachibana' method.")
    # Extract the harmonic component using Harmonic/Percussive Source Separation
    harmonic = librosa.effects.hpss(audio)[0]
    # Estimate the pitch of the harmonic part using Melodia
    return _extract_melodia(harmonic, sample_rate, hop_length)


def _extract_poliner(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Poliner & Ellis (2006) style STFT peak-tracking melody extraction."""
    if librosa is None:
        raise ImportError("librosa is required for 'poliner' method.")
    
    stft = np.abs(librosa.stft(audio, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    
    # Filter frequencies to typical vocal/melody range (80 Hz to 2000 Hz)
    valid_idx = np.where((frequencies >= 80) & (frequencies <= 2000))[0]
    if len(valid_idx) == 0:
        return np.full(stft.shape[1], np.nan), np.zeros(stft.shape[1])
        
    pitch_midi = []
    confidence = []
    
    for col in range(stft.shape[1]):
        frame_mag = stft[:, col]
        max_idx = valid_idx[np.argmax(frame_mag[valid_idx])]
        freq = frequencies[max_idx]
        pitch_midi.append(librosa.hz_to_midi(freq))
        # Confidence is ratio of max magnitude to frame sum
        confidence.append(frame_mag[max_idx] / (np.sum(frame_mag) + 1e-6))
        
    return np.array(pitch_midi), np.array(confidence)


def _extract_durrieu(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Durrieu's Source-Filter NMF approximation (NMF source separation + YIN)."""
    if librosa is None:
        raise ImportError("librosa is required for 'durrieu' method.")
    
    # Compute STFT magnitude spectrogram
    S = np.abs(librosa.stft(audio, hop_length=hop_length))
    
    # Run NMF to separate into lead/vocals (1) and accompaniment (0) components
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=2, init='random', random_state=0, max_iter=30)
    W = nmf.fit_transform(S)
    H = nmf.components_
    
    # Identify the vocal/lead track (typically has higher time-variance)
    idx = np.argmax(np.var(H, axis=1))
    lead_S = np.outer(W[:, idx], H[idx, :])
    
    # Reconstruct lead audio using Griffin-Lim
    lead_audio = librosa.griffinlim(lead_S, hop_length=hop_length)
    
    # Use YIN to estimate the pitch of the lead signal
    return _extract_yin(lead_audio, sample_rate, hop_length, 65.0, 2000.0, 2048)


def _extract_demucs_crepe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Demucs (vocals) + CREPE extraction."""
    if torch is None or get_model is None:
        raise ImportError(
            "torch and demucs are required for 'demucs_crepe' method but not installed."
        )

    # Prepare audio for Demucs: [channels, samples]
    # htdemucs expects stereo (2 channels)
    if audio.ndim == 1:
        audio_t = torch.from_numpy(audio).float()
        audio_t = torch.stack([audio_t, audio_t], dim=0)
    else:
        audio_t = torch.from_numpy(audio).float()
        if audio_t.shape[0] == 1:
            audio_t = torch.cat([audio_t, audio_t], dim=0)

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Load model (htdemucs is the standard high-quality model)
    if "demucs" not in _MODEL_CACHE:
        print("Cargando modelo Demucs...")
        model = get_model("htdemucs")
        model.to(device)
        model.eval()
        _MODEL_CACHE["demucs"] = model
    else:
        model = _MODEL_CACHE["demucs"]

    # Mixture T
    audio_t = audio_t.to(device)

    # Cache check
    audio_hash = _get_audio_hash(audio)
    cache_path = _get_vocal_cache_path(audio_hash, "htdemucs")
    
    if cache_path.exists():
        print(f"Loading extracted vocals from cache: {cache_path.name}")
        vocals_np = np.load(cache_path)
    else:
        print(f"Running Demucs source separation for {label}..." if label else "Running Demucs source separation...")
        # apply_model returns [sources, channels, samples]
        with torch.no_grad():
            sources = apply_model(model, audio_t[None])[0]

        # Get vocals index
        try:
            vocal_idx = model.sources.index("vocals")
        except ValueError:
            vocal_idx = 3

        vocals_t = sources[vocal_idx]
        # Back to mono and numpy
        vocals_np = vocals_t.mean(0).cpu().numpy()
        np.save(cache_path, vocals_np)

    # Run CREPE on isolated vocals
    return _extract_crepe(vocals_np, sample_rate, hop_length, label=label)


def _get_vocal_cache_path(audio_hash: str, method: str) -> Path:
    """Generate a path for caching extracted vocals on disk."""
    cache_dir = Path(__file__).parent / ".vocal_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{method}_{audio_hash}.npy"

def _get_audio_hash(audio: np.ndarray) -> str:
    """Compute a stable hash for an audio array."""
    # Only hash a portion to be fast but relatively unique
    prefix = audio[:10000].tobytes()
    return hashlib.md5(prefix + str(len(audio)).encode()).hexdigest()


def _get_bs_roformer_vocals(
    audio: np.ndarray,
    sample_rate: int,
    label: str = "",
) -> np.ndarray:
    """Helper to separate vocals using BS-Roformer (with caching)."""
    try:
        import bs_roformer
        from bs_roformer.model_registry import MODEL_REGISTRY
        from bs_roformer.utils import get_model_from_config, demix_track
        from bs_roformer.download import download_model_assets
        import torch
        # Monkeypatch torch.cuda.amp.autocast for Mac MPS speedup and warning suppression
        if torch.backends.mps.is_available():
            import torch.amp
            torch.cuda.amp.autocast = lambda enabled=True, dtype=torch.float16, cache_enabled=None: torch.amp.autocast("mps", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)
            
        import yaml
        from ml_collections import ConfigDict
    except ImportError as e:
        raise ImportError(
            f"bs-roformer-infer, onnxruntime, and ml-collections are required for BS-Roformer. Error: {e}"
        )

    # Setup Model Path
    models_base_dir = Path(__file__).parent / "models"
    bs_model_dir = models_base_dir / "bs_roformer"
    model_slug = "roformer-model-bs-roformer-sw-by-jarredou"
    
    try:
        model_info = MODEL_REGISTRY.get(model_slug)
    except KeyError:
        available = MODEL_REGISTRY.list("vocals")
        if not available:
            raise RuntimeError("No BS-Roformer vocal models found in registry.")
        model_info = available[0]
        model_slug = model_info.slug

    ckpt_path = bs_model_dir / model_slug / model_info.checkpoint
    config_path = bs_model_dir / model_slug / model_info.config

    if not ckpt_path.exists() or not config_path.exists():
        print(f"Downloading BS-Roformer model: {model_info.name}...")
        download_model_assets([model_info], bs_model_dir)

    # Load Config and Model
    if f"bs_roformer_{model_slug}" not in _MODEL_CACHE:
        print(f"Cargando modelo BS-Roformer: {model_info.name}...")
        with open(config_path) as f:
            config_data = yaml.load(f, Loader=yaml.FullLoader)
            config = ConfigDict(config_data)

        model = get_model_from_config("bs_roformer", config)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model.to(device)
        model.eval()
        _MODEL_CACHE[f"bs_roformer_{model_slug}"] = (model, config)
    else:
        model, config = _MODEL_CACHE[f"bs_roformer_{model_slug}"]
        device = next(model.parameters()).device

    # Prepare audio for separation (expects [channels, samples])
    if audio.ndim == 1:
        mix = np.stack([audio, audio], axis=0) # Stereo
    else:
        mix = audio
        if mix.shape[0] > 2: # Transpose if [samples, channels]
            mix = mix.T
        if mix.shape[0] == 1:
            mix = np.repeat(mix, 2, axis=0)

    # Mixture T
    mixture_t = torch.from_numpy(mix).float().to(device)

    # Cache check
    audio_hash = _get_audio_hash(audio)
    cache_path = _get_vocal_cache_path(audio_hash, f"bs_roformer_{model_slug}")
    
    if cache_path.exists():
        print(f"Loading extracted vocals from cache: {cache_path.name}")
        vocals_np = np.load(cache_path)
    else:
        if device.type == "mps":
            print("[Dispositivos] CUDA no está disponible en Mac (comportamiento normal). Utilizando GPU de Apple Silicon (MPS) para máxima aceleración...")
        elif device.type == "cuda":
            print("[Dispositivos] Utilizando GPU de Nvidia (CUDA)...")
        else:
            print("[Dispositivos] GPU no disponible. Utilizando procesador (CPU)...")
            
        print(f"Running BS-Roformer source separation for {label} (this may take a while)..." if label else "Running BS-Roformer source separation (this may take a while)...")
        with torch.no_grad():
            autocast_device = "cuda" if device.type == "cuda" else None
            if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch, "autocast"):
                 autocast_device = "mps"
            
            if autocast_device:
                with torch.autocast(device_type=autocast_device):
                    res, _ = demix_track(config, model, mixture_t, device)
            else:
                res, _ = demix_track(config, model, mixture_t, device)

        # Cleanup BS-Roformer mixture tensor immediately
        del mixture_t
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Isolated vocals (mono)
        vocals_np = res['vocals'].mean(0)
        
        # Cleanup results dictionary
        del res
        gc.collect()
        
        np.save(cache_path, vocals_np)

    return vocals_np

def _extract_bs_roformer_rmvpe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for BS-RoFormer (vocals) + RMVPE extraction."""
    if interp1d is None:
        raise ImportError("scipy is required for 'bs_roformer_rmvpe' method.")

    vocals_np = _get_bs_roformer_vocals(audio, sample_rate, label=label)

    # Pitch Extraction with RMVPE
    print(f"Running RMVPE pitch extraction for {label}..." if label else "Running RMVPE pitch extraction...")
    models_base_dir = Path(__file__).parent / "models"
    rmvpe_model_path = models_base_dir / "rmvpe.onnx"
    
    # Use cached RMVPE instance if available
    from .rmvpe_onnx import RMVPE
    if "rmvpe_instance" not in _MODEL_CACHE:
        _MODEL_CACHE["rmvpe_instance"] = RMVPE(str(rmvpe_model_path))
    
    rmvpe_model = _MODEL_CACHE["rmvpe_instance"]
    f0, conf = rmvpe_model.infer(vocals_np, sample_rate)

    # Cleanup vocals array after extraction
    del vocals_np
    gc.collect()

    # Align and Convert
    pitch_midi = np.zeros_like(f0)
    voiced = f0 > 0
    pitch_midi[voiced] = librosa.hz_to_midi(f0[voiced])
    pitch_midi[~voiced] = 0.0 # Use 0 for interpolation

    rmvpe_hop_s = 160 / 16000.0
    rmvpe_times = np.arange(len(pitch_midi)) * rmvpe_hop_s

    n_target_frames = int(np.ceil(len(audio) / hop_length))
    target_times = librosa.frames_to_time(
        np.arange(n_target_frames),
        sr=sample_rate,
        hop_length=hop_length
    )

    f_pitch = interp1d(rmvpe_times, pitch_midi, kind='linear', fill_value="extrapolate", bounds_error=False)
    f_conf = interp1d(rmvpe_times, conf, kind='linear', fill_value="extrapolate", bounds_error=False)

    pitch_interp = f_pitch(target_times)
    conf_interp = f_conf(target_times)

    pitch_interp[conf_interp < 0.2] = np.nan

    return pitch_interp, conf_interp


def _extract_bs_roformer_crepe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for BS-RoFormer (vocals) + CREPE extraction."""
    vocals_np = _get_bs_roformer_vocals(audio, sample_rate, label=label)
    return _extract_crepe(vocals_np, sample_rate, hop_length, label=label)


def _extract_demucs_rmvpe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Demucs (vocals) + RMVPE extraction."""
    if torch is None or get_model is None:
        raise ImportError(
            "torch and demucs are required for 'demucs_rmvpe' method but not installed."
        )

    # Prepare audio for Demucs
    if audio.ndim == 1:
        audio_t = torch.from_numpy(audio).float()
        audio_t = torch.stack([audio_t, audio_t], dim=0)
    else:
        audio_t = torch.from_numpy(audio).float()
        if audio_t.shape[0] == 1:
            audio_t = torch.cat([audio_t, audio_t], dim=0)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    if "demucs" not in _MODEL_CACHE:
        print("Cargando modelo Demucs...")
        model = get_model("htdemucs")
        model.to(device)
        model.eval()
        _MODEL_CACHE["demucs"] = model
    else:
        model = _MODEL_CACHE["demucs"]

    audio_t = audio_t.to(device)
    audio_hash = _get_audio_hash(audio)
    cache_path = _get_vocal_cache_path(audio_hash, "htdemucs")
    
    if cache_path.exists():
        print(f"Loading extracted vocals from cache: {cache_path.name}")
        vocals_np = np.load(cache_path)
    else:
        print(f"Running Demucs source separation for {label}..." if label else "Running Demucs source separation...")
        with torch.no_grad():
            sources = apply_model(model, audio_t[None])[0]

        try:
            vocal_idx = model.sources.index("vocals")
        except ValueError:
            vocal_idx = 3

        vocals_t = sources[vocal_idx]
        vocals_np = vocals_t.mean(0).cpu().numpy()
        np.save(cache_path, vocals_np)

    return _extract_rmvpe(vocals_np, sample_rate, hop_length, label=label)


def _extract_basic_pitch(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Spotify's Basic Pitch extraction."""
    try:
        from basic_pitch.inference import predict
        import tempfile
        import soundfile as sf
        import os
    except ImportError as e:
        raise ImportError(
            f"basic-pitch, soundfile are required for 'basic_pitch' method. Install them with 'pip install basic-pitch soundfile'. Error: {e}"
        )

    # Calculate target times based on original audio length and sample rate/hop_length
    n_target_frames = int(np.ceil(len(audio) / hop_length))
    target_times = librosa.frames_to_time(
        np.arange(n_target_frames),
        sr=sample_rate,
        hop_length=hop_length
    )

    # Now resample audio to 22050 Hz if necessary for basic-pitch
    audio_bp = audio
    bp_sr = 22050
    if sample_rate != 22050:
        if librosa is not None:
            audio_bp = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
        else:
            raise RuntimeError("librosa is required for resampling in basic-pitch extraction.")

    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio_bp, bp_sr)
        print(f"Running Basic Pitch inference for {label}..." if label else "Running Basic Pitch inference...")
        model_output, midi_data, note_events = predict(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    notes = []
    for inst in midi_data.instruments:
        notes.extend(inst.notes)
    
    # Sort notes by pitch ascending, so higher pitch notes overwrite lower ones (soprano priority)
    notes.sort(key=lambda n: n.pitch)
    
    pitch_midi = np.zeros_like(target_times)
    confidence = np.zeros_like(target_times)
    
    for note in notes:
        start_idx = np.searchsorted(target_times, note.start)
        end_idx = np.searchsorted(target_times, note.end)
        
        if start_idx < len(target_times):
            pitch_midi[start_idx:end_idx] = note.pitch
            confidence[start_idx:end_idx] = note.velocity / 127.0
            
            if start_idx == end_idx:
                pitch_midi[start_idx] = note.pitch
                confidence[start_idx] = note.velocity / 127.0

    pitch_midi[confidence == 0] = np.nan
    
    return pitch_midi, confidence


def _extract_rmvpe(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for standalone RMVPE extraction."""
    try:
        from .rmvpe_onnx import get_rmvpe_f0
        from scipy.interpolate import interp1d
        from pathlib import Path
    except ImportError as e:
        raise ImportError(f"onnxruntime, scipy, and Path are required for 'rmvpe' method. Error: {e}")

    models_base_dir = Path(__file__).parent / "models"
    rmvpe_model_path = models_base_dir / "rmvpe.onnx"
    
    if not rmvpe_model_path.exists():
         raise FileNotFoundError(f"RMVPE model not found at {rmvpe_model_path}")

    # Run RMVPE
    from .rmvpe_onnx import RMVPE
    if "rmvpe" not in _MODEL_CACHE:
        print("Cargando modelo RMVPE...")
        _MODEL_CACHE["rmvpe"] = RMVPE(str(rmvpe_model_path))
    
    rmvpe_model = _MODEL_CACHE["rmvpe"]
    f0, conf = rmvpe_model.infer(audio, sample_rate)

    # Align and Convert (RMVPE is 10ms/160 samples at 16kHz)
    pitch_midi = np.zeros_like(f0)
    voiced = f0 > 0
    pitch_midi[voiced] = librosa.hz_to_midi(f0[voiced])
    pitch_midi[~voiced] = 0.0

    rmvpe_hop_s = 160 / 16000.0
    rmvpe_times = np.arange(len(pitch_midi)) * rmvpe_hop_s
    
    n_target_frames = int(np.ceil(len(audio) / hop_length))
    target_times = librosa.frames_to_time(np.arange(n_target_frames), sr=sample_rate, hop_length=hop_length)

    f_pitch = interp1d(rmvpe_times, pitch_midi, kind='linear', fill_value="extrapolate", bounds_error=False)
    f_conf = interp1d(rmvpe_times, conf, kind='linear', fill_value="extrapolate", bounds_error=False)

    return f_pitch(target_times), f_conf(target_times)


def _extract_spice(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for Google's SPICE (via TensorFlow Hub)."""
    if hub is None:
        raise ImportError(
            "tensorflow_hub is required for 'spice' method. Install it with 'pip install tensorflow-hub'"
        )

    import tensorflow as tf

    # SPICE expects 16kHz mono audio
    if sample_rate != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio

    # Load model from TF Hub
    if "spice" not in _MODEL_CACHE:
        print("Loading SPICE model from TensorFlow Hub...")
        _MODEL_CACHE["spice"] = hub.load("https://tfhub.dev/google/spice/2")
    
    model = _MODEL_CACHE["spice"]

    # Audio should be in range [-1, 1], and float32
    audio_t = tf.constant(audio_16k, dtype=tf.float32)
    
    # Model returns pitch and uncertainty
    # SPICE Hub module is not directly callable, use the serving_default signature
    output = model.signatures["serving_default"](audio_t)
    pitch_outputs = output["pitch"]
    uncertainty_outputs = output["uncertainty"]

    # Convert pitch (0..1) to Hz
    # SPICE pitch is normalized: Hz = 2^(pitch * 6.35 + 1.66) * 110
    # Or something similar. Actually SPICE output 0.5 = 220Hz? 
    # The actual formula from TF Hub: frequency = 2.0 ** (pitch * 6.35 + 1.66) * 110.0
    # Wait, the official formula is: 
    # pitch_hz = exp(pitch * (log(fmax) - log(fmin)) + log(fmin))
    # where fmin=31.1Hz, fmax=2000Hz (6 octaves)
    fmin, fmax = 31.11, 1975.53 # Approx C1 to B6
    pitch_hz = np.exp(pitch_outputs.numpy() * (np.log(fmax) - np.log(fmin)) + np.log(fmin))
    
    # Confidence from uncertainty (uncertainty is 0..1, 0 is certain)
    confidence = 1.0 - uncertainty_outputs.numpy()
    
    # SPICE has a fixed hop size of 512 at 16kHz (32ms)
    spice_hop_s = 512 / 16000.0
    spice_times = np.arange(len(pitch_hz)) * spice_hop_s
    
    pitch_midi = librosa.hz_to_midi(pitch_hz)
    
    # Interpolate to target
    n_target_frames = int(np.ceil(len(audio) / hop_length))
    target_times = librosa.frames_to_time(np.arange(n_target_frames), sr=sample_rate, hop_length=hop_length)
    
    f_pitch = interp1d(spice_times, pitch_midi, kind='linear', fill_value="extrapolate", bounds_error=False)
    f_conf = interp1d(spice_times, confidence, kind='linear', fill_value="extrapolate", bounds_error=False)
    
    return f_pitch(target_times), f_conf(target_times)


def _extract_jdc(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for JDC (Joint Detection and Classification) vocal pitch estimator."""
    raise NotImplementedError(
        "JDC (Joint Detection and Classification) vocal pitch estimator is not installed or configured in this environment. "
        "Please check the installation of the vocal-pitch-estimator package."
    )


def _extract_fcn_f0(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Helper for FCN-f0 estimator."""
    import sys
    from pathlib import Path

    fcn_dir = Path(__file__).parent / "fcn_f0_src"
    if not fcn_dir.exists():
        raise ImportError(
            "FCN-f0 source directory not found. Please clone the FCN-f0 repository to src/melody_analysis_v2/fcn_f0_src"
        )

    # Add FCN-f0 directory to path to allow import of models, prediction, etc.
    if str(fcn_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(fcn_dir.resolve()))

    try:
        from models.load_model import load_model
        from prediction import sliding_norm, predict_fullConv
    except ImportError as e:
        raise ImportError(
            f"Failed to import modules from FCN-f0. Ensure fcn_f0_src is correct. Error: {e}"
        )

    # FCN-f0 expects mono audio at 8000 Hz.
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Resample audio to 8000 Hz
    if sample_rate != 8000:
        audio_8k = librosa.resample(audio, orig_sr=sample_rate, target_sr=8000)
    else:
        audio_8k = audio

    model_tag = "993"
    model_input_size = 993
    model_srate = 8000.0

    model_key = f"fcn_{model_tag}"
    if model_key not in _MODEL_CACHE:
        print(f"Cargando modelo FCN-f0 (tag: {model_tag})...")
        # Load in FULLCONV mode, inputSize is None
        _MODEL_CACHE[model_key] = load_model(model_tag, FULLCONV=True)

    model = _MODEL_CACHE[model_key]

    # Pad so that frames are centered (like in get_audio of prediction.py)
    padded_audio = np.pad(audio_8k, int(model_input_size // 2), mode='constant', constant_values=0)

    # Sliding norm
    norm_audio = sliding_norm(padded_audio, frame_sizes=model_input_size)
    norm_audio = np.reshape(norm_audio, (len(norm_audio), 1, 1))
    norm_audio = np.array([norm_audio])

    # Run prediction
    print(f"Running FCN-f0 inference for {label}..." if label else "Running FCN-f0 inference...")
    (timeVec, frequencies, confidence, activations) = predict_fullConv(model, norm_audio, viterbi=False, model_srate=model_srate)

    # Convert frequencies to MIDI note numbers
    pitch_midi = np.zeros_like(frequencies)
    voiced = frequencies > 0
    pitch_midi[voiced] = librosa.hz_to_midi(frequencies[voiced])
    pitch_midi[~voiced] = 0.0  # Set unvoiced to 0 for interpolation

    # Align and interpolate to target timestamps
    n_target_frames = int(np.ceil(len(audio) / hop_length))
    target_times = librosa.frames_to_time(
        np.arange(n_target_frames),
        sr=sample_rate,
        hop_length=hop_length
    )

    f_pitch = interp1d(timeVec, pitch_midi, kind='linear', fill_value="extrapolate", bounds_error=False)
    f_conf = interp1d(timeVec, confidence, kind='linear', fill_value="extrapolate", bounds_error=False)

    pitch_interp = f_pitch(target_times)
    conf_interp = f_conf(target_times)

    # Voicing detection: set unvoiced frames to NaN
    pitch_interp[conf_interp < 0.2] = np.nan

    return pitch_interp, conf_interp


def extract_melody_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    method: str = "pyin",
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 1000.0,
    frame_length: int = 2048,
    label: str = "",
) -> MelodyFeatures:
    """Estimate melody features from a raw audio signal.

    Parameters
    ----------
    audio:
        Audio samples.
    sample_rate:
        Sampling rate of ``audio``.
    method:
        Extraction method: 'pyin', 'crepe', 'ensemble', 'melodia', 'demucs_crepe', 'rmvpe', or 'bs_roformer_rmvpe'.
    hop_length:
        Hop length used for feature extraction.
    fmin, fmax:
        Frequency range used for pitch estimation (mainly for pyin).
    frame_length:
        Frame length employed by the RMS computation.

    Returns
    -------
    MelodyFeatures
        Extracted time, pitch, confidence, and energy trajectories.
    """

    if librosa is None:
        raise ImportError(
            "librosa is required for feature extraction but is not available."
        )

    audio = np.asarray(audio, dtype=float)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Normalize audio to avoid numerical issues with pitch extraction.
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Extract pitch and confidence based on method
    print(f"[{method}] Running pitch extraction for {label}..." if label else f"[{method}] Running pitch extraction...")
    if method == "pyin":
        pitch_midi, confidence = _extract_pyin(
            audio, sample_rate, hop_length, fmin, fmax, frame_length
        )
    elif method == "yin":
        pitch_midi, confidence = _extract_yin(
            audio, sample_rate, hop_length, fmin, fmax, frame_length
        )
    elif method == "crepe":
        pitch_midi, confidence = _extract_crepe(audio, sample_rate, hop_length, label=label)
    elif method == "melodia":
        pitch_midi, confidence = _extract_melodia(audio, sample_rate, hop_length)
    elif method == "tachibana":
        pitch_midi, confidence = _extract_tachibana(audio, sample_rate, hop_length)
    elif method == "poliner":
        pitch_midi, confidence = _extract_poliner(audio, sample_rate, hop_length)
    elif method == "durrieu":
        pitch_midi, confidence = _extract_durrieu(audio, sample_rate, hop_length)
    elif method == "demucs_crepe":
        pitch_midi, confidence = _extract_demucs_crepe(audio, sample_rate, hop_length, label=label)
    elif method == "ensemble":
        p_midi, p_conf = _extract_pyin(
            audio, sample_rate, hop_length, fmin, fmax, frame_length
        )
        c_midi, c_conf = _extract_crepe(audio, sample_rate, hop_length, label=label)

        # Basic alignment check (crepe and librosa might differ by 1 frame depending on padding)
        min_len = min(len(p_midi), len(c_midi))
        p_midi, p_conf = p_midi[:min_len], p_conf[:min_len]
        c_midi, c_conf = c_midi[:min_len], c_conf[:min_len]

        # Ensemble: Choose the one with higher confidence for each frame
        # Or a weighted average? Let's go with max confidence for pitch selection
        pitch_midi = np.where(p_conf >= c_conf, p_midi, c_midi)
        confidence = np.maximum(p_conf, c_conf)
    elif method == "bs_roformer_rmvpe" or method == "bs_roformer":
        pitch_midi, confidence = _extract_bs_roformer_rmvpe(audio, sample_rate, hop_length, label=label)
    elif method == "bs_roformer_crepe":
        pitch_midi, confidence = _extract_bs_roformer_crepe(audio, sample_rate, hop_length, label=label)
    elif method == "demucs_crepe" or method == "demucs":
        pitch_midi, confidence = _extract_demucs_crepe(audio, sample_rate, hop_length, label=label)
    elif method == "demucs_rmvpe":
        pitch_midi, confidence = _extract_demucs_rmvpe(audio, sample_rate, hop_length, label=label)
    elif method == "rmvpe":
        pitch_midi, confidence = _extract_rmvpe(audio, sample_rate, hop_length, label=label)
    elif method == "basic_pitch":
        pitch_midi, confidence = _extract_basic_pitch(audio, sample_rate, hop_length, label=label)
    elif method == "jdc":
        pitch_midi, confidence = _extract_jdc(audio, sample_rate, hop_length, label=label)
    elif method == "spice":
        pitch_midi, confidence = _extract_spice(audio, sample_rate, hop_length)
    elif method == "fcn_f0":
        pitch_midi, confidence = _extract_fcn_f0(audio, sample_rate, hop_length, label=label)
    else:
        raise ValueError(f"Unknown melody extraction method: {method}")

    # Interpolate unvoiced frames for the MIDI contour
    # (Note: we keep the confidence low so the classifier/segmenter knows it's unvoiced)
    pitch_midi = _interpolate_nans(pitch_midi)

    # Compute energy using RMS and align to pitch frames.
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    ).flatten()

    # Align energy length to pitch_midi length
    if len(energy) > len(pitch_midi):
        energy = energy[: len(pitch_midi)]
    elif len(energy) < len(pitch_midi):
        energy = np.pad(energy, (0, len(pitch_midi) - len(energy)), mode="edge")

    energy = _interpolate_nans(energy)
    if energy.max() > 0:
        energy = energy / energy.max()

    times = librosa.frames_to_time(
        np.arange(len(pitch_midi)),
        sr=sample_rate,
        hop_length=hop_length,
    )

    return MelodyFeatures(times=times, pitch_midi=pitch_midi, confidence=confidence, energy=energy)


__all__ = ["MelodyFeatures", "extract_melody_features"]
