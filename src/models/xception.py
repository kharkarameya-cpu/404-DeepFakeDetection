import timm
import torch.nn as nn


class XceptionBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0  # Remove classifier
        )

    def forward(self, x):
        return self.model(x)