"""Utilities for synthesizing audio from melody features."""

import numpy as np
import librosa

def synthesize_melody(
    times: np.ndarray,
    pitch_midi: np.ndarray,
    confidence: np.ndarray,
    energy: np.ndarray,
    sample_rate: int = 22050,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    """
    Synthesizes a sine wave from melody features.

    Parameters
    ----------
    times : np.ndarray
        Time stamps for each frame.
    pitch_midi : np.ndarray
        MIDI pitch values.
    confidence : np.ndarray
        Confidence values [0, 1].
    energy : np.ndarray
        Energy values [0, 1].
    sample_rate : int
        Output sampling rate.
    confidence_threshold : float
        Threshold below which the melody is silenced.

    Returns
    -------
    np.ndarray
        Synthesized audio signal.
    """
    if len(times) == 0:
        return np.array([], dtype=float)

    duration = times[-1]
    n_samples = int(np.ceil(duration * sample_rate))
    
    # Target time grid for synthesis
    t_audio = np.arange(n_samples) / sample_rate
    
    # Interpolate pitch (MIDI) and control signals to audio sample rate
    # We use MIDI interpolation to avoid frequency artifacts during transitions
    pitch_interp = np.interp(t_audio, times, pitch_midi)
    conf_interp = np.interp(t_audio, times, confidence)
    energy_interp = np.interp(t_audio, times, energy)
    
    # Convert MIDI to Hz
    freq_hz = librosa.midi_to_hz(pitch_interp)
    
    # Silence unvoiced frames
    freq_hz[conf_interp < confidence_threshold] = 0.0
    
    # Phase accumulation for continuous sine wave (avoids clicks)
    # phase[n] = phase[n-1] + 2 * pi * f[n] / sr
    phases = np.cumsum(2.0 * np.pi * freq_hz / sample_rate)
    
    # Generate sine wave
    audio = np.sin(phases)
    
    # Apply energy envelope and confidence gating
    audio *= energy_interp
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
        
    return audio
