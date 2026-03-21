"""Data preparation script for BirdCLEF+ 2026.

Loads competition metadata, analyzes class distributions,
and creates stratified train/validation splits.
"""

import os
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_data(data_dir="data", val_split=0.2, seed=42, min_rating=3.0):
    """Prepare training data splits.

    Args:
        data_dir: Root data directory containing train.csv, taxonomy.csv, etc.
        val_split: Fraction of data for validation.
        seed: Random seed.
        min_rating: Minimum quality rating to keep.
    """
    # Load metadata
    train_csv = os.path.join(data_dir, "train.csv")
    taxonomy_csv = os.path.join(data_dir, "taxonomy.csv")

    train_df = pd.read_csv(train_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)

    print("=" * 60)
    print("BirdCLEF+ 2026 Data Preparation")
    print("=" * 60)

    # Basic stats
    print(f"\nTotal recordings: {len(train_df)}")
    print(f"Unique species: {train_df['primary_label'].nunique()}")
    print(f"Species in taxonomy: {len(taxonomy_df)}")

    if "rating" in train_df.columns:
        print(f"\nRating distribution:")
        print(train_df["rating"].describe())

    # Class distribution
    class_counts = train_df["primary_label"].value_counts()
    print(f"\nClass distribution:")
    print(f"  Max recordings per species: {class_counts.max()} ({class_counts.idxmax()})")
    print(f"  Min recordings per species: {class_counts.min()} ({class_counts.idxmin()})")
    print(f"  Median recordings per species: {class_counts.median():.0f}")
    print(f"  Imbalance ratio (max/min): {class_counts.max() / max(class_counts.min(), 1):.1f}")

    # Filter by quality
    if min_rating > 0 and "rating" in train_df.columns:
        before = len(train_df)
        train_df = train_df[train_df["rating"] >= min_rating].reset_index(drop=True)
        print(f"\nFiltered by rating >= {min_rating}: {before} -> {len(train_df)} recordings")

    # Stratified split
    train_split, val_split_df = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df["primary_label"],
        random_state=seed,
    )

    print(f"\nTrain split: {len(train_split)} recordings")
    print(f"Val split:   {len(val_split_df)} recordings")

    # Verify class balance
    train_classes = train_split["primary_label"].nunique()
    val_classes = val_split_df["primary_label"].nunique()
    print(f"Classes in train: {train_classes}")
    print(f"Classes in val:   {val_classes}")

    # Save splits
    train_out = os.path.join(data_dir, "train_split.csv")
    val_out = os.path.join(data_dir, "val_split.csv")
    train_split.to_csv(train_out, index=False)
    val_split_df.to_csv(val_out, index=False)
    print(f"\nSaved: {train_out}")
    print(f"Saved: {val_out}")

    # Soundscape labels stats
    soundscape_csv = os.path.join(data_dir, "train_soundscapes_labels.csv")
    if os.path.exists(soundscape_csv):
        sc_df = pd.read_csv(soundscape_csv)
        print(f"\nSoundscape annotations: {len(sc_df)}")
        print(f"Unique soundscape files: {sc_df['filename'].nunique()}")
        print(f"Species in soundscapes: {sc_df['primary_label'].nunique()}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_rating", type=float, default=3.0, help="Minimum quality rating")
    args = parser.parse_args()
    prepare_data(args.data_dir, args.val_split, args.seed, args.min_rating)
