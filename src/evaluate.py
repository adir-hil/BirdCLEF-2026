"""Evaluation metrics for BirdCLEF+ 2026."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_roc_auc_per_class(y_true, y_scores, species_list):
    """Compute per-class and macro ROC-AUC.

    Skips classes with no positive samples (matches competition metric).

    Args:
        y_true: numpy array of shape (N, num_classes) with binary labels.
        y_scores: numpy array of shape (N, num_classes) with predicted probabilities.
        species_list: List of species names for labeling.

    Returns:
        Tuple of (per_class_auc dict, macro_auc float).
    """
    per_class_auc = {}
    valid_aucs = []

    for i, species in enumerate(species_list):
        if y_true[:, i].sum() == 0:
            continue
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            per_class_auc[species] = auc
            valid_aucs.append(auc)
        except ValueError:
            continue

    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    return per_class_auc, macro_auc


@torch.no_grad()
def evaluate_roc_auc(model, data_loader, device, species_list=None, model_type="simple"):
    """Evaluate model on a DataLoader and return macro ROC-AUC.

    Args:
        model: PyTorch model.
        data_loader: DataLoader yielding (spectrograms, labels) batches.
        device: torch device.
        species_list: Optional list of species names.
        model_type: 'simple', 'v2', or 'sed' to determine output format.

    Returns:
        macro ROC-AUC score (float).
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in data_loader:
        spectrograms, labels = batch[0], batch[1]
        spectrograms = spectrograms.to(device)

        output = model(spectrograms)

        # Handle different model output formats
        if isinstance(output, dict):
            logits = output["clipwise_logits"]
        else:
            logits = output

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.numpy()

        all_preds.append(probs)
        all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if species_list is None:
        species_list = [str(i) for i in range(all_preds.shape[1])]

    _, macro_auc = compute_roc_auc_per_class(all_targets, all_preds, species_list)
    return macro_auc


@torch.no_grad()
def get_predictions(model, data_loader, device, model_type="simple"):
    """Get raw predictions from a model.

    Returns:
        Tuple of (all_probs, all_row_ids_or_labels)
    """
    model.eval()
    all_probs = []
    all_ids = []

    for batch in data_loader:
        spectrograms = batch[0].to(device)

        output = model(spectrograms)
        if isinstance(output, dict):
            logits = output["clipwise_logits"]
        else:
            logits = output

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

        # batch[1] could be labels (tensor) or row_ids (list of strings)
        if isinstance(batch[1], torch.Tensor):
            all_ids.append(batch[1].numpy())
        else:
            all_ids.extend(batch[1])

    all_probs = np.concatenate(all_probs, axis=0)

    if isinstance(all_ids[0], np.ndarray):
        all_ids = np.concatenate(all_ids, axis=0)

    return all_probs, all_ids
