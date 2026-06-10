import numpy as np
import gc
import librosa
import onnxruntime as ort

class RMVPE:
    def __init__(self, model_path):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
        # RMVPE parameters
        self.sample_rate = 16000
        self.hop_length = 160
        self.n_mel = 128
        self.n_fft = 1024
        self.win_length = 1024
        
        # Frequency to Cent conversion
        # RMVPE uses 360 bins for 0 to 4500 cents (12.5 cents per bin)
        self.cents_per_bin = 12.5
        self.n_bins = 360
        self.fmin_hz = 32.703 # C1
        
    def _preprocess(self, audio):
        """Preprocess audio to log-mel spectrogram."""
        # Ensure 16kHz
        # librosa.stft, then mel
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel,
            fmin=self.fmin_hz,
            center=True,
            pad_mode='reflect'
        )
        # Log mel
        S_log = librosa.power_to_db(S, ref=np.max, top_db=80)
        del S
        gc.collect()
        
        # Normalize to [-1, 1] loosely or as expected by model
        # RMVPE usually expects specific normalization. 
        # Actually, many RVC implementations use:
        # (S_log + 40) / 40
        S_norm = (S_log + 40) / 40
        return S_norm.astype(np.float32)

    def infer(self, audio, sr):
        """Predict f0 from audio signal using memory-efficient chunked inference with fixed input shapes."""
        # 1. Resample to 16kHz
        if sr != self.sample_rate:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            # Use 16k version for mel
            mel = self._preprocess(audio_16k)
            del audio_16k
        else:
            mel = self._preprocess(audio)
        
        gc.collect()
        
        input_name = self.session.get_inputs()[0].name
        n_frames = mel.shape[1]
        
        # Chunking parameters (fixed size 1024)
        chunk_size = 1024
        overlap = 128
        step = chunk_size - 2 * overlap # 768
        
        total_prob = np.zeros((n_frames, 360), dtype=np.float32)
        
        # Pad mel with overlap on both sides
        mel_padded = np.pad(mel, ((0, 0), (overlap, overlap)), mode='constant')
        del mel
        gc.collect()
        
        for start in range(0, n_frames, step):
            # Extract a chunk of size exactly chunk_size
            chunk = mel_padded[:, start : start + chunk_size]
            
            # If chunk size is smaller than chunk_size, pad it
            if chunk.shape[1] < chunk_size:
                pad_width = chunk_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0, 0), (0, pad_width)), mode='constant')
            
            # Invariant: chunk.shape is always (128, 1024)
            mel_input = chunk[np.newaxis, :, :]
            outputs = self.session.run(None, {input_name: mel_input})
            prob = outputs[0][0] # shape [1024, 360]
            
            write_start = start
            write_end = min(n_frames, start + step)
            read_start = overlap
            read_end = overlap + (write_end - write_start)
            
            total_prob[write_start:write_end, :] = prob[read_start:read_end, :]
            
        del mel_padded
        gc.collect()
        
        # 4. Postprocess
        # Argmax to get cents
        best_bins = np.argmax(total_prob, axis=-1)
        conf = np.max(total_prob, axis=-1)
        
        f0 = np.zeros_like(conf)
        voiced = conf > 0.3 # Threshold
        
        cents = best_bins[voiced] * self.cents_per_bin
        f0[voiced] = self.fmin_hz * (2 ** (cents / 1200))
        
        return f0, conf

def get_rmvpe_f0(audio, sr, model_path):
    rmvpe = RMVPE(model_path)
    return rmvpe.infer(audio, sr)
