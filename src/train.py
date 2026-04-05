"""Training loop for BirdCLEF+ 2026."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.audio import load_audio
from src.dataset import BirdCLEFDataset, SoundscapeDataset, load_taxonomy
from src.model import get_model
from src.transforms import get_audio_transforms, get_mixup_fn
from src.evaluate import evaluate_roc_auc
from src.utils import set_seed


def train_one_epoch(model, optimizer, data_loader, device, criterion, scaler=None, mixup_fn=None):
    """Train for one epoch.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc="Training")
    for batch in pbar:
        spectrograms, labels = batch[0].to(device), batch[1].to(device)

        # Apply mixup
        if mixup_fn is not None and np.random.random() < 0.5:
            spectrograms, labels = mixup_fn(spectrograms, labels)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                output = model(spectrograms)
                logits = output["clipwise_logits"] if isinstance(output, dict) else output
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(spectrograms)
            logits = output["clipwise_logits"] if isinstance(output, dict) else output
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

    return running_loss / max(num_batches, 1)


def train(config_path="config/default.yaml"):
    """Main training function.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Best validation ROC-AUC score.
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load taxonomy
    species_list = load_taxonomy(config["data"]["taxonomy_csv"])
    print(f"Number of species: {len(species_list)}")

    # Load training metadata
    train_df = pd.read_csv(config["data"]["train_csv"])
    print(f"Total recordings: {len(train_df)}")

    # Filter by quality rating
    min_rating = config["data"].get("min_rating", 0)
    if min_rating > 0 and "rating" in train_df.columns:
        train_df = train_df[train_df["rating"] >= min_rating].reset_index(drop=True)
        print(f"After rating filter (>= {min_rating}): {len(train_df)}")

    # Train/val split
    val_split = config["data"].get("val_split", 0.2)
    train_split, val_split_df = train_test_split(
        train_df,
        test_size=val_split,
        stratify=train_df["primary_label"],
        random_state=config["data"]["seed"],
    )
    print(f"Train: {len(train_split)}, Val: {len(val_split_df)}")

    # Audio transforms (training only)
    audio_transforms = get_audio_transforms(config.get("augmentation", {}))

    # Create datasets
    train_dataset = BirdCLEFDataset(
        df=train_split,
        audio_dir=config["data"]["train_audio_dir"],
        config=config,
        species_list=species_list,
        audio_transforms=audio_transforms,
        is_train=True,
    )
    val_dataset = BirdCLEFDataset(
        df=val_split_df,
        audio_dir=config["data"]["train_audio_dir"],
        config=config,
        species_list=species_list,
        audio_transforms=None,
        is_train=False,
    )

    # Optionally add soundscape data to training
    soundscape_labels_path = config["data"].get("soundscape_labels_csv")
    soundscape_dir = config["data"].get("train_soundscapes_dir")
    if soundscape_labels_path and os.path.exists(soundscape_labels_path):
        labels_df = pd.read_csv(soundscape_labels_path)
        soundscape_dataset = SoundscapeDataset(
            soundscape_dir=soundscape_dir,
            config=config,
            species_list=species_list,
            labels_df=labels_df,
            is_test=False,
        )
        train_dataset = ConcatDataset([train_dataset, soundscape_dataset])
        print(f"Added {len(soundscape_dataset)} soundscape windows to training")

    # DataLoaders
    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    model = get_model(config["model"]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
    )

    # Loss
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None,
        reduction="mean",
    )

    # Mixup
    mixup_fn = None
    aug_config = config.get("augmentation", {})
    if aug_config.get("mixup_alpha", 0) > 0:
        mixup_fn = get_mixup_fn(aug_config["mixup_alpha"])

    # AMP scaler
    scaler = GradScaler() if train_cfg.get("amp", False) and device.type == "cuda" else None

    # Training loop
    os.makedirs("models", exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Warmup: linearly increase LR
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        avg_loss = train_one_epoch(
            model, optimizer, train_loader, device, criterion,
            scaler=scaler, mixup_fn=mixup_fn,
        )

        if epoch > warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Validation
        val_interval = train_cfg.get("val_interval", 1)
        if epoch % val_interval == 0:
            val_auc = evaluate_roc_auc(model, val_loader, device, species_list)
            print(f"Val ROC-AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), "models/best_model.pth")
                print(f"Saved best model (ROC-AUC: {best_auc:.4f})")

    print(f"\nTraining complete. Best Val ROC-AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
