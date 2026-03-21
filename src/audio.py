"""Audio loading and mel spectrogram conversion."""

import numpy as np
import librosa


def load_audio(path, sr=32000, duration=5.0, offset=0.0):
    """Load an audio file, resample, and extract a fixed-length segment.

    Args:
        path: Path to audio file (OGG, WAV, etc.)
        sr: Target sample rate in Hz.
        duration: Duration in seconds to extract.
        offset: Start time in seconds.

    Returns:
        numpy array of shape (num_samples,) with float32 values.
    """
    num_samples = int(sr * duration)

    try:
        import torchaudio
        # First load metadata to get sample rate for offset calculation
        info = torchaudio.info(path)
        orig_sr = info.sample_rate
        frame_offset = int(offset * orig_sr) if offset > 0 else 0
        num_frames = int(duration * orig_sr)
        waveform, orig_sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        audio = waveform[0].numpy()
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        audio, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)

    # Pad or truncate to fixed length
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)), mode="constant")
    else:
        audio = audio[:num_samples]

    return audio.astype(np.float32)


def audio_to_melspec(audio, sr, config):
    """Convert waveform to log-mel spectrogram.

    Args:
        audio: numpy array of shape (num_samples,).
        sr: Sample rate in Hz.
        config: Dict with keys: n_mels, n_fft, hop_length, fmin, fmax, power, top_db.

    Returns:
        numpy array of shape (n_mels, time_frames) in dB scale.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=config.get("n_mels", 128),
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512),
        fmin=config.get("fmin", 50),
        fmax=config.get("fmax", 14000),
        power=config.get("power", 2.0),
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=config.get("top_db", 80))
    return mel_db.astype(np.float32)


def load_as_melspec(path, config, offset=0.0):
    """Load audio file and convert to mel spectrogram in one step.

    Args:
        path: Path to audio file.
        config: Dict with audio parameters (sample_rate, duration, n_mels, etc.)
        offset: Start time in seconds.

    Returns:
        numpy array of shape (n_mels, time_frames).
    """
    sr = config.get("sample_rate", 32000)
    duration = config.get("duration", 5.0)
    audio = load_audio(path, sr=sr, duration=duration, offset=offset)
    return audio_to_melspec(audio, sr, config)


def normalize_melspec(melspec):
    """Normalize spectrogram to zero mean, unit variance.

    Args:
        melspec: numpy array of shape (n_mels, time_frames).

    Returns:
        Normalized spectrogram (same shape).
    """
    mean = melspec.mean()
    std = melspec.std()
    if std > 0:
        melspec = (melspec - mean) / std
    else:
        melspec = melspec - mean
    return melspec
