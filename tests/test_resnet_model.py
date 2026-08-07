"""Tests for ResNetTransferLearning model architecture and parameter groups."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from src.models.resnet_model import ResNetTransferLearning


@pytest.fixture
def resnet18():
    return ResNetTransferLearning(architecture='resnet18', num_classes=8,
                                  pretrained=False, freeze_backbone=False)


@pytest.fixture
def resnet18_frozen():
    return ResNetTransferLearning(architecture='resnet18', num_classes=8,
                                  pretrained=False, freeze_backbone=True)


def test_model_constructs(resnet18):
    assert resnet18 is not None


def test_model_forward_shape(resnet18):
    x = torch.rand(2, 3, 224, 224)
    out = resnet18(x)
    assert out.shape == (2, 8)


def test_model_num_classes_4(resnet18):
    """Output dim matches num_classes param."""
    m = ResNetTransferLearning(architecture='resnet18', num_classes=4,
                               pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    assert m(x).shape == (1, 4)


def test_freeze_backbone_sets_no_grad(resnet18_frozen):
    trainable = [n for n, p in resnet18_frozen.model.named_parameters()
                 if p.requires_grad and 'fc' not in n]
    assert len(trainable) == 0


def test_unfreeze_all_makes_trainable(resnet18_frozen):
    resnet18_frozen.unfreeze_all()
    frozen = [p for p in resnet18_frozen.parameters() if not p.requires_grad]
    assert len(frozen) == 0


def test_unfreeze_last_block_layer4_trainable(resnet18_frozen):
    resnet18_frozen.unfreeze_last_block()
    layer4_trainable = [n for n, p in resnet18_frozen.model.named_parameters()
                        if 'layer4' in n and p.requires_grad]
    assert len(layer4_trainable) > 0


def test_param_groups_structure(resnet18_frozen):
    resnet18_frozen.unfreeze_all()
    groups = resnet18_frozen.get_parameter_groups()
    # Must have at least classifier group
    names = [g['name'] for g in groups]
    assert 'classifier' in names


def test_param_groups_lrs_differ(resnet18_frozen):
    resnet18_frozen.unfreeze_all()
    groups = resnet18_frozen.get_parameter_groups(
        lr_backbone=1e-5, lr_layer4=1e-4, lr_classifier=1e-3)
    lrs = [g['lr'] for g in groups]
    assert len(set(lrs)) > 1  # discriminative LRs must differ


def test_extract_features_shape(resnet18):
    x = torch.rand(3, 3, 224, 224)
    feats = resnet18.extract_features(x)
    assert feats.ndim == 2
    assert feats.shape[0] == 3
    assert feats.shape[1] == 512  # ResNet18 feature dim


def test_model_no_nan_output(resnet18):
    x = torch.rand(4, 3, 224, 224)
    out = resnet18(x)
    assert not torch.isnan(out).any()
