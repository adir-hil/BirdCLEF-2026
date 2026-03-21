"""Audio and spectrogram augmentations for BirdCLEF+ 2026.

Includes:
- Waveform augmentations (audiomentations)
- SpecAugment (frequency & time masking)
- Mixup / CutMix for batch-level augmentation
- Spectrogram normalization and conversion
"""

import numpy as np
import torch


def get_audio_transforms(config, train=True):
    """Create waveform-level augmentation pipeline.

    Args:
        config: Dict with augmentation parameters.
        train: Whether this is for training (applies augmentations) or val/test.

    Returns:
        audiomentations.Compose pipeline, or None if unavailable/not training.
    """
    if not train:
        return None

    try:
        from audiomentations import (
            Compose, AddGaussianNoise, TimeStretch, PitchShift,
            Shift, Gain, AddColoredNoise, LowPassFilter, HighPassFilter,
        )

        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015,
                             p=config.get("add_noise_prob", 0.3)),
            AddColoredNoise(min_snr_db=3, max_snr_db=30, p=0.2),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=0.2, leave_length_unchanged=True),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
            Shift(min_shift=-0.5, max_shift=0.5, p=config.get("time_shift_prob", 0.3)),
            Gain(min_gain_db=-12, max_gain_db=12, p=config.get("gain_prob", 0.3)),
            LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=7500, p=0.1),
            HighPassFilter(min_cutoff_freq=20, max_cutoff_freq=500, p=0.1),
        ])
    except ImportError:
        return None


def spec_augment(melspec, config):
    """Apply SpecAugment (frequency and time masking) to a mel spectrogram.

    Args:
        melspec: numpy array of shape (n_mels, time_frames).
        config: Dict with keys: freq_mask_param, time_mask_param,
                num_freq_masks, num_time_masks.

    Returns:
        Augmented spectrogram (same shape).
    """
    spec = melspec.copy()
    n_mels, n_frames = spec.shape

    freq_mask_param = config.get("freq_mask_param", 20)
    time_mask_param = config.get("time_mask_param", 40)
    num_freq_masks = config.get("num_freq_masks", 2)
    num_time_masks = config.get("num_time_masks", 2)

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels))
        f0 = np.random.randint(0, max(1, n_mels - f))
        spec[f0:f0 + f, :] = 0

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, n_frames))
        t0 = np.random.randint(0, max(1, n_frames - t))
        spec[:, t0:t0 + t] = 0

    return spec


def get_mixup_fn(alpha=0.5):
    """Create a batch-level mixup function.

    Args:
        alpha: Beta distribution parameter.

    Returns:
        Callable that takes (spectrograms, labels) tensors and returns mixed versions.
    """
    def mixup(spectrograms, labels):
        lam = np.random.beta(alpha, alpha)
        batch_size = spectrograms.size(0)
        index = torch.randperm(batch_size, device=spectrograms.device)
        mixed_spec = lam * spectrograms + (1 - lam) * spectrograms[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        return mixed_spec, mixed_labels, lam

    return mixup


def get_cutmix_fn(alpha=1.0):
    """Create a batch-level CutMix function for spectrograms.

    Cuts a rectangular region from one sample and pastes it onto another.
    Particularly effective for audio since it simulates overlapping sounds.
    """
    def cutmix(spectrograms, labels):
        lam = np.random.beta(alpha, alpha)
        batch_size = spectrograms.size(0)
        index = torch.randperm(batch_size, device=spectrograms.device)

        _, _, h, w = spectrograms.shape

        # Random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)

        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)

        mixed = spectrograms.clone()
        mixed[:, :, y1:y2, x1:x2] = spectrograms[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual area
        lam_adjusted = 1 - (y2 - y1) * (x2 - x1) / (h * w)
        mixed_labels = lam_adjusted * labels + (1 - lam_adjusted) * labels[index]

        return mixed, mixed_labels, lam_adjusted

    return cutmix


def spectrogram_to_tensor(melspec, in_channels=1):
    """Convert mel spectrogram to a PyTorch tensor.

    Args:
        melspec: numpy array of shape (n_mels, time_frames).
        in_channels: 1 for mono, 3 for pseudo-RGB (repeat channels).

    Returns:
        torch.FloatTensor of shape (in_channels, n_mels, time_frames).
    """
    if in_channels == 3:
        tensor = np.stack([melspec] * 3, axis=0)
    else:
        tensor = melspec[np.newaxis, :]

    return torch.from_numpy(tensor).float()
