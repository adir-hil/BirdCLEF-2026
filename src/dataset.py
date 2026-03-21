"""PyTorch datasets for BirdCLEF+ 2026."""

import os
import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.audio import load_audio, audio_to_melspec, normalize_melspec
from src.transforms import spec_augment, spectrogram_to_tensor


def load_taxonomy(csv_path):
    """Load species list from taxonomy.csv.

    Returns:
        List of species code strings (234 entries).
    """
    df = pd.read_csv(csv_path)
    if "primary_label" in df.columns:
        return df["primary_label"].tolist()
    # Fallback: use first column that looks like species codes
    return df.iloc[:, 0].tolist()


def build_label_vector(primary_label, secondary_labels, species_to_idx, num_classes=234):
    """Create a multi-hot label vector.

    Args:
        primary_label: String species code.
        secondary_labels: List of string species codes, or string repr of list.
        species_to_idx: Dict mapping species code to index.
        num_classes: Total number of classes.

    Returns:
        numpy array of shape (num_classes,) with float32 values.
    """
    labels = np.zeros(num_classes, dtype=np.float32)

    if primary_label in species_to_idx:
        labels[species_to_idx[primary_label]] = 1.0

    if isinstance(secondary_labels, str):
        try:
            secondary_labels = ast.literal_eval(secondary_labels)
        except (ValueError, SyntaxError):
            secondary_labels = []

    if isinstance(secondary_labels, list):
        for label in secondary_labels:
            if label in species_to_idx:
                labels[species_to_idx[label]] = 1.0

    return labels


class BirdCLEFDataset(Dataset):
    """Dataset for individual training recordings from train_audio/."""

    def __init__(self, df, audio_dir, config, species_list, audio_transforms=None, is_train=True):
        """
        Args:
            df: DataFrame with columns: filename, primary_label, secondary_labels, rating.
            audio_dir: Path to train_audio/ directory.
            config: Full config dict (with 'audio', 'augmentation', 'model' sections).
            species_list: List of 234 species codes.
            audio_transforms: Optional audiomentations pipeline for waveform augmentation.
            is_train: Whether to apply augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.audio_config = config.get("audio", {})
        self.aug_config = config.get("augmentation", {})
        self.in_channels = config.get("model", {}).get("in_channels", 1)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.audio_transforms = audio_transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])

        sr = self.audio_config.get("sample_rate", 32000)
        duration = self.audio_config.get("duration", 5.0)

        # Random offset for training (crop augmentation)
        offset = 0.0
        if self.is_train:
            try:
                import librosa
                total_dur = librosa.get_duration(path=file_path)
                if total_dur > duration:
                    offset = np.random.uniform(0, total_dur - duration)
            except Exception:
                pass

        # Load audio
        audio = load_audio(file_path, sr=sr, duration=duration, offset=offset)

        # Waveform augmentation
        if self.is_train and self.audio_transforms is not None:
            audio = self.audio_transforms(samples=audio, sample_rate=sr)

        # Convert to mel spectrogram
        melspec = audio_to_melspec(audio, sr, self.audio_config)
        melspec = normalize_melspec(melspec)

        # Spectrogram augmentation (SpecAugment)
        if self.is_train and self.aug_config.get("spec_augment", False):
            melspec = spec_augment(melspec, self.aug_config)

        # Convert to tensor
        tensor = spectrogram_to_tensor(melspec, self.in_channels)

        # Build label vector
        labels = build_label_vector(
            row["primary_label"],
            row.get("secondary_labels", "[]"),
            self.species_to_idx,
            self.num_classes,
        )

        return tensor, torch.from_numpy(labels)


class SoundscapeDataset(Dataset):
    """Dataset for continuous soundscape recordings."""

    def __init__(self, soundscape_dir, config, species_list, labels_df=None, is_test=False):
        """
        Args:
            soundscape_dir: Path to soundscape directory.
            config: Full config dict.
            species_list: List of 234 species codes.
            labels_df: DataFrame with columns: filename, start, end, primary_label.
                       None for test set.
            is_test: If True, generate all windows without labels.
        """
        self.soundscape_dir = soundscape_dir
        self.audio_config = config.get("audio", {})
        self.in_channels = config.get("model", {}).get("in_channels", 1)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.is_test = is_test
        self.duration = self.audio_config.get("duration", 5.0)

        # Build index of all 5-second windows
        self.windows = []
        self._build_window_index(soundscape_dir, labels_df)

    def _build_window_index(self, soundscape_dir, labels_df):
        """Pre-compute all (file_path, start_sec, label_vector, row_id) entries."""
        import librosa

        files = sorted([
            f for f in os.listdir(soundscape_dir)
            if f.endswith(".ogg")
        ])

        for filename in files:
            file_path = os.path.join(soundscape_dir, filename)
            total_dur = librosa.get_duration(path=file_path)
            basename = os.path.splitext(filename)[0]

            start = 0.0
            while start + self.duration <= total_dur + 0.01:
                end = start + self.duration
                row_id = f"{basename}_{int(end)}"

                if self.is_test:
                    label_vec = None
                else:
                    label_vec = np.zeros(self.num_classes, dtype=np.float32)
                    if labels_df is not None:
                        mask = (
                            (labels_df["filename"] == filename) &
                            (labels_df["start"] < end) &
                            (labels_df["end"] > start)
                        )
                        for _, lrow in labels_df[mask].iterrows():
                            species = lrow["primary_label"]
                            if species in self.species_to_idx:
                                label_vec[self.species_to_idx[species]] = 1.0

                self.windows.append({
                    "file_path": file_path,
                    "start": start,
                    "label_vec": label_vec,
                    "row_id": row_id,
                })
                start += self.duration

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        sr = self.audio_config.get("sample_rate", 32000)

        audio = load_audio(w["file_path"], sr=sr, duration=self.duration, offset=w["start"])
        melspec = audio_to_melspec(audio, sr, self.audio_config)
        melspec = normalize_melspec(melspec)
        tensor = spectrogram_to_tensor(melspec, self.in_channels)

        if self.is_test:
            return tensor, w["row_id"]

        return tensor, torch.from_numpy(w["label_vec"])
