import os
from typing import Optional

import torch
from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.utils.seg_xml_utils import rasterize_positive_mask, mask_to_box


class EndoMultiTaskDataset(EndoClassificationDataset):
    """
    Extends your classification dataset:
      metadata additions per sample:
        - 'gt_mask': HxW uint8 tensor or None (positive-only)
        - 'box': (4,) xyxy float32 or None
        - 'orig_hw': (H, W) tuple
    """
    def __getitem__(self, idx: int):
        image, label, meta = super().__getitem__(idx)

        gt_mask_t: Optional[torch.Tensor] = None
        box_t: Optional[torch.Tensor] = None

        xml_path = meta.get("xml")
        if xml_path and os.path.exists(xml_path):
            mask_np = rasterize_positive_mask(xml_path)  # HxW uint8
            H, W = mask_np.shape
            meta["orig_hw"] = (H, W)
            if mask_np.sum() > 0:
                gt_mask_t = torch.from_numpy(mask_np)  # HxW
                xyxy = mask_to_box(mask_np)
                if xyxy is not None:
                    box_t = torch.tensor(xyxy, dtype=torch.float32)
        else:
            # we still record a plausible orig size using image tensor
            if hasattr(image, "shape"):
                meta["orig_hw"] = (image.shape[-2], image.shape[-1])

        meta["gt_mask"] = gt_mask_t
        meta["box"] = box_t
        return image, label, meta