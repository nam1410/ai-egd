import timm
import torch
import torch.nn as nn
from typing import List
import os

from safetensors.torch import load_file as safe_load

def create_dinov2_classifier(num_classes: int = 2, model_name: str = 'vit_small_patch14_dinov2') -> nn.Module:
    """
    Create a DINOv2 classifier using timm.
    """
    model_path = os.path.expanduser("/lustre06/project/6103394/ofarooq/ai-egd/hub/vit_small_patch14_dinov2.lvd142m.safetensors")
    if os.path.exists(model_path):
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        state_dict = safe_load(model_path, device="cpu")  # Load safetensors safely
        model.load_state_dict(state_dict, strict=False)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module):
    # Freeze everything except the classifier head (called 'head' or 'fc' depending on timm model).
    for n, p in model.named_parameters():
        p.requires_grad = False
    # Try to unfreeze classifier head params
    head = getattr(model, 'head', None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True
    # Some timm ViTs use 'head' only; if custom heads exist, add logic here.


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2):
    """
    Gradually unfreeze last N transformer blocks to fine-tune.
    For ViT in timm, blocks are typically in model.blocks list.
    """
    blocks: List[nn.Module] = getattr(model, 'blocks', None)
    if blocks is None:
        return  # fallback if architecture differs
    # Unfreeze last n_blocks
    for blk in blocks[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    # Also ensure normalization layer before head is trainable
    for attr in ['norm', 'ln1', 'ln2']:
        m = getattr(model, attr, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True