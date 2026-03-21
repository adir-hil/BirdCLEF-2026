"""Utility functions for BirdCLEF+ 2026."""

import os
import random
import time

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
    """Create a submission row_id from filename and end time."""
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


class Timer:
    """Simple context-manager timer for tracking runtime budget."""

    def __init__(self, budget_seconds=5400):  # 90 minutes default
        self.budget = budget_seconds
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def remaining(self):
        return max(0, self.budget - self.elapsed())

    def check(self, required_seconds=60, msg=""):
        """Raise if not enough time remaining."""
        if self.remaining() < required_seconds:
            raise TimeoutError(
                f"Only {self.remaining():.0f}s remaining, need {required_seconds}s. {msg}"
            )

    def __str__(self):
        e = self.elapsed()
        r = self.remaining()
        return f"Elapsed: {e:.0f}s | Remaining: {r:.0f}s ({r/60:.1f}min)"


def ensemble_predictions(pred_list, weights=None, method="mean"):
    """Combine predictions from multiple models.

    Args:
        pred_list: List of numpy arrays, each shape (N, num_classes).
        weights: Optional list of float weights (must sum to 1).
        method: 'mean', 'geometric', or 'rank'.

    Returns:
        numpy array of shape (N, num_classes).
    """
    if weights is None:
        weights = [1.0 / len(pred_list)] * len(pred_list)

    if method == "mean":
        return sum(w * p for w, p in zip(weights, pred_list))

    elif method == "geometric":
        log_preds = [w * np.log(np.clip(p, 1e-7, 1.0)) for w, p in zip(weights, pred_list)]
        return np.exp(sum(log_preds))

    elif method == "rank":
        from scipy.stats import rankdata
        ranked = []
        for p in pred_list:
            ranked_p = np.zeros_like(p)
            for col in range(p.shape[1]):
                ranked_p[:, col] = rankdata(p[:, col]) / len(p)
            ranked.append(ranked_p)
        return sum(w * r for w, r in zip(weights, ranked))

    else:
        raise ValueError(f"Unknown ensemble method: {method}")
