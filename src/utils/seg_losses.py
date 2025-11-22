import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from segment_anything.utils.transforms import ResizeLongestSide


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits/targets: (B,1,H,W)
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(1,2,3))
        den = (probs.pow(2).sum(dim=(1,2,3)) + targets.pow(2).sum(dim=(1,2,3)) + self.eps)
        return 1.0 - (num / den).mean()


def resize_mask_like_sam(mask: torch.Tensor, orig_hw: Tuple[int, int], img_size: int = 1024, out_size: int = 256) -> torch.Tensor:
    """
    mask: (H, W) uint8/float -> returns (1, out_size, out_size) float {0,1}
    Reproduces SAM's resize+pad geometry before downsampling to 256.
    """
    if mask.dtype != torch.float32:
        mask = mask.float()
    mask = mask.unsqueeze(0).unsqueeze(0)  # 1x1xH xW
    H, W = orig_hw
    tf = ResizeLongestSide(img_size)
    new_h, new_w = tf.get_preprocess_shape(H, W, long_side_length=img_size)
    mask_resized = F.interpolate(mask, size=(new_h, new_w), mode='nearest')
    pad_h, pad_w = img_size - new_h, img_size - new_w
    mask_padded = F.pad(mask_resized, (0, pad_w, 0, pad_h))  # 1x1xSxS
    low = F.interpolate(mask_padded, size=(out_size, out_size), mode='nearest')
    return low  # 1x256x256