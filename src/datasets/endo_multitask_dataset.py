import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.utils.seg_xml_utils import rasterize_positive_mask, mask_to_box


class EndoMultiTaskDataset(EndoClassificationDataset):
    """
    Extends classification dataset:
      - attaches 'gt_mask' (HxW uint8 tensor) and 'box' (4,) to metadata when XML exists and is positive
    """
    def __getitem__(self, idx: int):
        image, label, meta = super().__getitem__(idx)

        gt_mask_t: Optional[torch.Tensor] = None
        box_t: Optional[torch.Tensor] = None

        xml_path = meta.get("xml")
        if xml_path and os.path.exists(xml_path):
            mask_np = rasterize_positive_mask(xml_path)  # HxW uint8
            if mask_np.sum() > 0:
                gt_mask_t = torch.from_numpy(mask_np)  # HxW
                xyxy = mask_to_box(mask_np)
                if xyxy is not None:
                    box_t = torch.tensor(xyxy, dtype=torch.float32)

        meta = dict(meta)
        meta["gt_mask"] = gt_mask_t  # HxW or None
        meta["box"] = box_t          # (4,) or None

        return image, label, meta