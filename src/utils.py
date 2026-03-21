"""Utility functions for BirdCLEF+ 2026."""

import os
import random

import numpy as np
import pandas as pd
import torch
import librosa


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_soundscape_windows(file_path, duration=5.0, sr=32000):
    """Compute all 5-second windows for a soundscape file.

    Args:
        file_path: Path to soundscape OGG file.
        duration: Window duration in seconds.
        sr: Sample rate (used to compute total duration).

    Returns:
        List of (start_second, end_second) tuples.
    """
    total_duration = librosa.get_duration(path=file_path)
    windows = []
    start = 0.0
    while start + duration <= total_duration + 0.01:
        end = start + duration
        windows.append((start, end))
        start += duration
    return windows


def make_row_id(filename, end_second):
    """Create a submission row_id from filename and end time.

    Args:
        filename: Soundscape filename (without extension).
        end_second: End time of the 5-second window in seconds.

    Returns:
        String like 'BC2026_Test_0001_S05_20250227_010002_20'.
    """
    basename = os.path.splitext(filename)[0]
    return f"{basename}_{int(end_second)}"


def create_submission_df(predictions, species_list):
    """Build submission DataFrame from predictions.

    Args:
        predictions: List of (row_id, probability_array) tuples.
        species_list: List of 234 species codes (column names).

    Returns:
        pandas DataFrame with columns [row_id, species1, ..., species234].
    """
    rows = []
    for row_id, probs in predictions:
        row = {"row_id": row_id}
        for species, prob in zip(species_list, probs):
            row[species] = float(prob)
        rows.append(row)
    return pd.DataFrame(rows)
