"""Model factory for BirdCLEF+ 2026."""

import torch.nn as nn
import timm


class BirdCLEFModel(nn.Module):
    """EfficientNet backbone with linear classification head."""

    def __init__(self, backbone_name, num_classes, pretrained=True, in_channels=1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,  # remove default classifier
        )
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def get_model(config):
    """Factory function to create a BirdCLEF model from config.

    Args:
        config: Dict with keys: backbone, num_classes, pretrained, in_channels.

    Returns:
        BirdCLEFModel instance.
    """
    return BirdCLEFModel(
        backbone_name=config.get("backbone", "tf_efficientnet_b0_ns"),
        num_classes=config.get("num_classes", 234),
        pretrained=config.get("pretrained", True),
        in_channels=config.get("in_channels", 1),
    )
