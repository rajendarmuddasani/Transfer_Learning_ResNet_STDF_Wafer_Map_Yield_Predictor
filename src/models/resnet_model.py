"""
ResNet Model with Transfer Learning
Implements progressive fine-tuning for wafer-pattern classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List
import logging

logger = logging.getLogger(__name__)


class ResNetTransferLearning(nn.Module):
    """
    ResNet model with transfer learning for wafer map classification.
    Supports progressive fine-tuning with layer-wise learning rates.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 8,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Initialize ResNet transfer learning model.

        Args:
            architecture: resnet18 or resnet50
            num_classes: Number of output classes (defect types)
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers initially
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes

        # Load pretrained model
        if architecture == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.model = models.resnet18(weights=weights)
            else:
                self.model = models.resnet18(weights=None)
            in_features = 512
        elif architecture == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.model = models.resnet50(weights=weights)
            else:
                self.model = models.resnet50(weights=None)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Replace final classifier
        self.model.fc = nn.Linear(in_features, num_classes)

        # Initialize new classifier
        nn.init.xavier_uniform_(self.model.fc.weight)
        nn.init.zeros_(self.model.fc.bias)

        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            f"Initialized {architecture} with {num_classes} classes, "
            f"pretrained={pretrained}, freeze_backbone={freeze_backbone}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all layers except final classifier."""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:  # Don't freeze classifier
                param.requires_grad = False
        logger.info("Froze backbone layers")

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all layers")

    def unfreeze_last_block(self):
        """Unfreeze only the last residual block (layer4)."""
        # Freeze all first
        self.freeze_backbone()

        # Unfreeze layer4 and classifier
        for name, param in self.model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True

        logger.info("Unfroze layer4 and classifier")

    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_layer4: float = 1e-4,
        lr_classifier: float = 1e-3
    ) -> List[dict]:
        """
        Get parameter groups with discriminative learning rates.

        Args:
            lr_backbone: Learning rate for layer1-3
            lr_layer4: Learning rate for layer4
            lr_classifier: Learning rate for classifier

        Returns:
            List of parameter group dictionaries
        """
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'layer4' not in n and 'fc' not in n and p.requires_grad],
                'lr': lr_backbone,
                'name': 'backbone'
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'layer4' in n and p.requires_grad],
                'lr': lr_layer4,
                'name': 'layer4'
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'fc' in n and p.requires_grad],
                'lr': lr_classifier,
                'name': 'classifier'
            }
        ]

        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        return param_groups

    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (model without classifier)."""
        return nn.Sequential(*list(self.model.children())[:-1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input (before classifier).

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Features tensor (B, D) where D=512 (ResNet18) or 2048 (ResNet50)
        """
        features = self.get_feature_extractor()(x)
        return torch.flatten(features, 1)


class ResNetYieldRegressor(nn.Module):
    """
    Experimental yield-regression architecture; no confirmed trained artifact.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Initialize ResNet yield regressor.

        Args:
            architecture: resnet18 or resnet50
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers initially
        """
        super().__init__()

        self.classifier = ResNetTransferLearning(
            architecture=architecture,
            num_classes=1,  # Single output for yield
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation for 0-1 range."""
        logits = self.classifier(x)
        return torch.sigmoid(logits)  # Output in [0, 1] range


class ResNetMultiTask(nn.Module):
    """
    Experimental multi-task architecture; no confirmed trained artifact.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 8,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Initialize multi-task ResNet.

        Args:
            architecture: resnet18 or resnet50
            num_classes: Number of defect classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers initially
        """
        super().__init__()

        # Shared backbone
        if architecture == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
                self.backbone = models.resnet18(weights=weights)
            else:
                self.backbone = models.resnet18(weights=None)
            in_features = 512
        elif architecture == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50(weights=None)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Remove original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Task-specific heads
        self.defect_classifier = nn.Linear(in_features, num_classes)
        self.yield_regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        logger.info(f"Initialized multi-task {architecture} model")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.

        Returns:
            (defect_logits, yield_prediction)
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)

        defect_logits = self.defect_classifier(features)
        yield_pred = self.yield_regressor(features)

        return defect_logits, yield_pred


def create_model(
    architecture: str = "resnet18",
    num_classes: int = 8,
    task: str = "classification",
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Factory function to create model.

    Args:
        architecture: resnet18 or resnet50
        num_classes: Number of classes (for classification)
        task: classification, regression, or multitask
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone initially

    Returns:
        Model instance
    """
    if task == "classification":
        return ResNetTransferLearning(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif task == "regression":
        return ResNetYieldRegressor(
            architecture=architecture,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif task == "multitask":
        return ResNetMultiTask(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    # Test model creation
    model = create_model(architecture="resnet18", num_classes=8, task="classification")
    print(f"Model: {model.architecture}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
