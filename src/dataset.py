"""PyTorch datasets for BirdCLEF+ 2026.

Supports:
- Individual training recordings (train_audio/)
- Soundscape recordings with labels (train_soundscapes/)
- Test soundscapes for inference
- K-fold cross-validation splits
"""

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
    return df.iloc[:, 0].tolist()


def build_label_vector(primary_label, secondary_labels, species_to_idx, num_classes=234,
                       secondary_weight=0.5):
    """Create a multi-hot label vector.

    Args:
        primary_label: String species code, or semicolon-separated codes for soundscapes.
        secondary_labels: List of string species codes, or string repr of list.
        species_to_idx: Dict mapping species code to index.
        num_classes: Total number of classes.
        secondary_weight: Weight for secondary labels (soft label).

    Returns:
        numpy array of shape (num_classes,) with float32 values.
    """
    labels = np.zeros(num_classes, dtype=np.float32)

    # Handle semicolon-separated primary labels (soundscape format)
    if isinstance(primary_label, str) and ";" in primary_label:
        for code in primary_label.split(";"):
            code = code.strip()
            if code in species_to_idx:
                labels[species_to_idx[code]] = 1.0
    elif isinstance(primary_label, str) and primary_label in species_to_idx:
        labels[species_to_idx[primary_label]] = 1.0

    if isinstance(secondary_labels, str):
        try:
            secondary_labels = ast.literal_eval(secondary_labels)
        except (ValueError, SyntaxError):
            secondary_labels = []

    if isinstance(secondary_labels, list):
        for label in secondary_labels:
            if label in species_to_idx:
                labels[species_to_idx[label]] = max(labels[species_to_idx[label]],
                                                     secondary_weight)

    return labels


def create_kfold_splits(train_df, n_folds=5, seed=42):
    """Create stratified k-fold splits.

    Args:
        train_df: DataFrame with 'primary_label' column.
        n_folds: Number of folds.
        seed: Random seed.

    Returns:
        DataFrame with added 'fold' column (0 to n_folds-1).
    """
    from sklearn.model_selection import StratifiedKFold

    df = train_df.copy()
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["primary_label"])):
        df.loc[val_idx, "fold"] = fold

    return df


class BirdCLEFDataset(Dataset):
    """Dataset for individual training recordings from train_audio/."""

    def __init__(self, df, audio_dir, config, species_list, audio_transforms=None,
                 is_train=True, secondary_weight=0.5):
        """
        Args:
            df: DataFrame with columns: filename, primary_label, secondary_labels, rating.
            audio_dir: Path to train_audio/ directory.
            config: Full config dict (with 'audio', 'augmentation', 'model' sections).
            species_list: List of 234 species codes.
            audio_transforms: Optional audiomentations pipeline for waveform augmentation.
            is_train: Whether to apply augmentations.
            secondary_weight: Weight for secondary labels in label vector.
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
        self.secondary_weight = secondary_weight

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
            secondary_weight=self.secondary_weight,
        )

        return tensor, torch.from_numpy(labels)


class SoundscapeDataset(Dataset):
    """Dataset for continuous soundscape recordings with 5-second windows."""

    def __init__(self, soundscape_dir, config, species_list, labels_df=None,
                 is_test=False, augment=False, audio_transforms=None,
                 neg_ratio=3, seed=42):
        """
        Args:
            soundscape_dir: Path to soundscape directory.
            config: Full config dict.
            species_list: List of 234 species codes.
            labels_df: DataFrame with columns: filename, start, end, primary_label.
            is_test: If True, generate all windows without labels.
            augment: If True, apply SpecAugment.
            audio_transforms: Optional waveform augmentation pipeline.
            neg_ratio: Max ratio of negative (empty) to positive windows.
                       e.g. 3 means keep at most 3 negatives per positive.
                       Set to None or 0 to keep all windows (no downsampling).
            seed: Random seed for reproducible negative sampling.
        """
        self.soundscape_dir = soundscape_dir
        self.audio_config = config.get("audio", {})
        self.aug_config = config.get("augmentation", {})
        self.in_channels = config.get("model", {}).get("in_channels", 1)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.is_test = is_test
        self.augment = augment
        self.audio_transforms = audio_transforms
        self.duration = self.audio_config.get("duration", 5.0)

        self.windows = []
        self._build_window_index(soundscape_dir, labels_df)

        # Stratified downsampling: keep all positives, subsample negatives
        if not is_test and neg_ratio:
            self._downsample_negatives(neg_ratio, seed)

    def _build_window_index(self, soundscape_dir, labels_df):
        """Pre-compute all (file_path, start_sec, label_vector, row_id) entries."""
        import soundfile as sf

        # Ensure start/end columns are numeric (CSV may load them as strings)
        if labels_df is not None:
            labels_df = labels_df.copy()
            labels_df["start"] = pd.to_numeric(labels_df["start"], errors="coerce")
            labels_df["end"] = pd.to_numeric(labels_df["end"], errors="coerce")

        # Pre-group labels by filename for fast lookup (avoid repeated DataFrame filtering)
        labels_by_file = {}
        if labels_df is not None and not self.is_test:
            # Debug: show CSV structure
            print(f"  Labels CSV columns: {list(labels_df.columns)}")
            print(f"  Labels CSV sample filenames: {labels_df['filename'].unique()[:3].tolist()}")

            for filename, group in labels_df.groupby("filename"):
                labels_by_file[filename] = group[["start", "end", "primary_label"]].values

            # Also index by stem (without extension) for flexible matching
            for filename in list(labels_by_file.keys()):
                stem = os.path.splitext(filename)[0]
                if stem not in labels_by_file:
                    labels_by_file[stem] = labels_by_file[filename]

        all_files = sorted([f for f in os.listdir(soundscape_dir) if f.endswith(".ogg")])

        # For training: only process labeled files (skip thousands of unlabeled ones)
        if not self.is_test and labels_by_file:
            files = [f for f in all_files if f in labels_by_file or os.path.splitext(f)[0] in labels_by_file]
            print(f"  Soundscape files: {len(files)} labeled / {len(all_files)} total (skipping unlabeled)")
        else:
            files = all_files
            print(f"  Soundscape files: {len(files)} total")

        # Debug: check label time ranges and species for first labeled file
        if labels_by_file and not self.is_test:
            first_key = next(iter(labels_by_file))
            sample_labels = labels_by_file[first_key]
            print(f"  Sample label rows for '{first_key}':")
            for row in sample_labels[:5]:
                sp_in_list = all(
                    s.strip() in self.species_to_idx
                    for s in str(row[2]).split(";") if s.strip()
                )
                print(f"    start={row[0]}, end={row[1]}, species='{row[2]}', in_taxonomy={sp_in_list}")

        for filename in files:
            file_path = os.path.join(soundscape_dir, filename)
            # Use soundfile for fast duration check (reads header only, no decode)
            info = sf.info(file_path)
            total_dur = info.duration
            basename = os.path.splitext(filename)[0]

            # Get this file's labels once (try full name, then stem without extension)
            file_labels = labels_by_file.get(filename, None)
            if file_labels is None:
                file_labels = labels_by_file.get(basename, None)

            start = 0.0
            while start + self.duration <= total_dur + 0.01:
                end = start + self.duration
                row_id = f"{basename}_{int(end)}"

                if self.is_test:
                    label_vec = None
                else:
                    label_vec = np.zeros(self.num_classes, dtype=np.float32)
                    if file_labels is not None:
                        # Vectorized overlap check on numpy arrays
                        lbl_starts = file_labels[:, 0].astype(float)
                        lbl_ends = file_labels[:, 1].astype(float)
                        overlap = (lbl_starts < end) & (lbl_ends > start)
                        for row in file_labels[overlap]:
                            species_str = row[2]
                            if isinstance(species_str, str):
                                for sp in species_str.split(";"):
                                    sp = sp.strip()
                                    if sp in self.species_to_idx:
                                        label_vec[self.species_to_idx[sp]] = 1.0

                self.windows.append({
                    "file_path": file_path,
                    "start": start,
                    "label_vec": label_vec,
                    "row_id": row_id,
                })
                start += self.duration

    def _downsample_negatives(self, neg_ratio, seed):
        """Keep all positive windows, downsample negatives to neg_ratio:1."""
        positives = [w for w in self.windows if w["label_vec"].sum() > 0]
        negatives = [w for w in self.windows if w["label_vec"].sum() == 0]

        max_neg = len(positives) * neg_ratio
        if len(negatives) <= max_neg:
            return  # already balanced enough

        rng = np.random.RandomState(seed)
        sampled_neg = [negatives[i] for i in rng.choice(len(negatives), size=max_neg, replace=False)]
        self.windows = positives + sampled_neg
        rng.shuffle(self.windows)

        print(f"  Soundscape sampling: {len(positives)} positive + {len(sampled_neg)} negative "
              f"(from {len(negatives)} total neg, ratio 1:{neg_ratio})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        sr = self.audio_config.get("sample_rate", 32000)

        audio = load_audio(w["file_path"], sr=sr, duration=self.duration, offset=w["start"])

        # Waveform augmentation
        if self.augment and self.audio_transforms is not None:
            audio = self.audio_transforms(samples=audio, sample_rate=sr)

        melspec = audio_to_melspec(audio, sr, self.audio_config)
        melspec = normalize_melspec(melspec)

        # Spec augmentation
        if self.augment and self.aug_config.get("spec_augment", False):
            melspec = spec_augment(melspec, self.aug_config)

        tensor = spectrogram_to_tensor(melspec, self.in_channels)

        if self.is_test:
            return tensor, w["row_id"]

        return tensor, torch.from_numpy(w["label_vec"])


class BalancedSampler(torch.utils.data.Sampler):
    """Weighted random sampler that upsamples rare species.

    Critical for BirdCLEF where class imbalance is extreme.
    """

    def __init__(self, dataset_df, species_list):
        self.num_samples = len(dataset_df)
        class_counts = dataset_df["primary_label"].value_counts()

        # Weight = 1 / sqrt(count) for each sample's class
        weights = []
        for _, row in dataset_df.iterrows():
            count = class_counts.get(row["primary_label"], 1)
            weights.append(1.0 / np.sqrt(count))

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
