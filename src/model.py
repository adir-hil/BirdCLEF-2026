"""Model architectures for BirdCLEF+ 2026.

Includes:
- Simple classifier (BirdCLEFModel) for baseline
- SED model with attention pooling (BirdCLEFSED) for competitive submissions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BirdCLEFModel(nn.Module):
    """Simple EfficientNet backbone with linear classification head."""

    def __init__(self, backbone_name, num_classes, pretrained=True, in_channels=1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class AttentionPooling(nn.Module):
    """Attention-based pooling for Sound Event Detection.

    Instead of global average pooling, learns which time frames
    are most important for each class.
    """

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1),  # attention weights over time
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, time_frames, features)
        Returns:
            clipwise_logits: (batch, num_classes)
            segmentwise_logits: (batch, time_frames, num_classes)
        """
        # Segment-level predictions
        segmentwise_logits = self.classifier(x)  # (B, T, C)

        # Attention weights per class
        att_weights = self.attention(x)  # (B, T, C)

        # Weighted average of segment predictions
        clipwise_logits = (att_weights * segmentwise_logits).sum(dim=1)  # (B, C)

        return clipwise_logits, segmentwise_logits


class BirdCLEFSED(nn.Module):
    """Sound Event Detection model with attention pooling.

    This is the standard competitive BirdCLEF architecture:
    1. CNN backbone extracts features from mel spectrogram
    2. Features are pooled along frequency axis -> time-series of feature vectors
    3. Attention pooling learns which time frames matter per species
    4. Both clip-level and segment-level outputs available

    This handles the domain gap between clean training recordings and
    noisy continuous soundscapes much better than simple classification.
    """

    def __init__(self, backbone_name, num_classes, pretrained=True, in_channels=1,
                 dropout=0.3):
        super().__init__()

        # Create backbone without pooling/classifier
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,  # return feature maps, not pooled features
        )

        # Get the number of channels from the last feature map
        # We need to do a dummy forward pass to figure out the feature dim
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 128, 313)  # ~5s at 32kHz
            features = self.backbone(dummy)
            last_feat = features[-1]  # (B, C, H, W)
            self.feat_channels = last_feat.shape[1]
            self.feat_h = last_feat.shape[2]

        # Pool frequency dimension, keep time dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W) -> flatten -> (B, W, C)

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.feat_channels)

        # Attention-based pooling
        self.att_pool = AttentionPooling(self.feat_channels, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        """
        Args:
            x: (batch, channels, n_mels, time_frames)
        Returns:
            dict with keys:
                'clipwise_logits': (batch, num_classes) - for training/inference
                'segmentwise_logits': (batch, time_frames, num_classes) - optional
        """
        # Extract features
        features = self.backbone(x)
        feat = features[-1]  # last feature map: (B, C, H, W)

        # Pool frequency, keep time: (B, C, H, W) -> (B, C, 1, W) -> (B, C, W)
        feat = self.freq_pool(feat).squeeze(2)  # (B, C, W)

        # Transpose to (B, W, C) for attention
        feat = feat.transpose(1, 2)  # (B, T, C)

        # Apply BN on feature dim
        feat_bn = self.bn(feat.transpose(1, 2)).transpose(1, 2)
        feat_bn = self.dropout(feat_bn)

        # Attention pooling
        clipwise_logits, segmentwise_logits = self.att_pool(feat_bn)

        return {
            "clipwise_logits": clipwise_logits,
            "segmentwise_logits": segmentwise_logits,
        }


class GeMPooling(nn.Module):
    """Generalized Mean Pooling - better than average pooling for retrieval."""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class BirdCLEFModelV2(nn.Module):
    """Enhanced classifier with GeM pooling + multi-dropout for better generalization."""

    def __init__(self, backbone_name, num_classes, pretrained=True, in_channels=1,
                 dropout=0.3, num_dropouts=5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",  # no pooling - we use GeM
        )
        self.gem = GeMPooling()
        self.bn = nn.BatchNorm1d(self.backbone.num_features)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_dropouts)])
        self.head = nn.Linear(self.backbone.num_features, num_classes)
        self.num_dropouts = num_dropouts

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = self.gem(features)
        features = self.bn(features)

        if self.training:
            # Multi-sample dropout for better training
            logits = sum(self.head(d(features)) for d in self.dropouts) / self.num_dropouts
        else:
            logits = self.head(features)

        return logits


def get_model(config):
    """Factory function to create a model from config.

    Args:
        config: Dict with keys:
            - backbone: timm model name
            - num_classes: number of output classes
            - pretrained: whether to use pretrained weights
            - in_channels: 1 for mono, 3 for RGB
            - model_type: 'simple', 'sed', or 'v2' (default: 'simple')
            - dropout: dropout rate (for SED/V2)

    Returns:
        Model instance.
    """
    model_type = config.get("model_type", "simple")
    backbone = config.get("backbone", "tf_efficientnet_b0_ns")
    num_classes = config.get("num_classes", 234)
    pretrained = config.get("pretrained", True)
    in_channels = config.get("in_channels", 1)
    dropout = config.get("dropout", 0.3)

    if model_type == "sed":
        return BirdCLEFSED(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout=dropout,
        )
    elif model_type == "v2":
        return BirdCLEFModelV2(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
            dropout=dropout,
        )
    else:
        return BirdCLEFModel(
            backbone_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            in_channels=in_channels,
        )
