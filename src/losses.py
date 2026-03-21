"""Loss functions for BirdCLEF+ 2026.

Includes:
- Focal Loss for handling class imbalance
- BCE with label smoothing
- Asymmetric Loss for multi-label classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    Down-weights easy examples and focuses on hard ones.
    Critical for BirdCLEF where most species are absent (easy negatives).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BCEWithSmoothing(nn.Module):
    """BCE loss with label smoothing for better calibration."""

    def __init__(self, smoothing=0.025, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            pos_weight=self.pos_weight,
        )


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Different gamma for positive and negative examples.
    Allows stronger focusing on hard positives while reducing
    the contribution of easy negatives.

    Reference: https://arxiv.org/abs/2009.14119
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, reduction="mean"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Asymmetric clipping
        probs_neg = (probs + self.clip).clamp(max=1)

        # Basic CE
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt_pos = probs
            pt_neg = 1 - probs
            one_sided_gamma_pos = torch.pow(1 - pt_pos, self.gamma_pos)
            one_sided_gamma_neg = torch.pow(pt_neg, self.gamma_neg)
            loss_pos = loss_pos * one_sided_gamma_pos
            loss_neg = loss_neg * one_sided_gamma_neg

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_criterion(config):
    """Factory function for loss functions.

    Args:
        config: Dict with keys:
            - loss_type: 'bce', 'focal', 'asymmetric', 'bce_smooth'
            - label_smoothing: smoothing factor for BCE
            - focal_alpha: alpha for focal loss
            - focal_gamma: gamma for focal loss

    Returns:
        Loss function module.
    """
    loss_type = config.get("loss_type", "bce")

    if loss_type == "focal":
        return FocalLoss(
            alpha=config.get("focal_alpha", 0.25),
            gamma=config.get("focal_gamma", 2.0),
        )
    elif loss_type == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=config.get("asl_gamma_neg", 4),
            gamma_pos=config.get("asl_gamma_pos", 1),
        )
    elif loss_type == "bce_smooth":
        return BCEWithSmoothing(
            smoothing=config.get("label_smoothing", 0.025),
        )
    else:
        return nn.BCEWithLogitsLoss()
