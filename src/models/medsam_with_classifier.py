import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import numpy as np

# pip install git+https://github.com/facebookresearch/segment-anything
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class MedSamWithClassifier(nn.Module):
    """
    Wrap SAM/MedSAM and add a classification head.

    Returns:
      cls_logits: (B, num_classes)
      seg_logits: Optional[(B, 1, 256, 256)] — present only if boxes are provided
    """
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",    # 'vit_b'|'vit_l'|'vit_h' — must match your checkpoint
        num_classes: int = 2,
        freeze_image_encoder: bool = True,
        use_mask_guided_pool: bool = False,  # if True, concat ROI pooled features when a mask is available
    ):
        super().__init__()
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.use_mask_guided_pool = use_mask_guided_pool

        # Preprocess
        self.img_size = self.sam.image_encoder.img_size  # typically 1024
        self.transform = ResizeLongestSide(self.img_size)
        self.register_buffer("pixel_mean", torch.tensor(self.sam.pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(self.sam.pixel_std).view(1, 3, 1, 1), persistent=False)

        # Classification head on encoder features (B, 256, H', W')
        cls_in = 256
        cls_dim = cls_in if not use_mask_guided_pool else (cls_in * 2)  # global + ROI pooled
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_dim),
            nn.Linear(cls_dim, num_classes),
        )

        if freeze_image_encoder:
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _preprocess_batch_images(self, images: List[np.ndarray]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        batched = []
        orig_sizes = []
        for img in images:
            h, w = img.shape[:2]
            orig_sizes.append((h, w))
            resized = self.transform.apply_image(img)  # H' x W' x 3
            x = torch.as_tensor(resized, dtype=torch.float32).permute(2, 0, 1)
            x = (x - self.pixel_mean) / self.pixel_std
            pad_h, pad_w = self.img_size - x.shape[1], self.img_size - x.shape[2]
            x = F.pad(x, (0, pad_w, 0, pad_h))
            batched.append(x)
        return torch.stack(batched, dim=0), orig_sizes

    def _ensure_numpy_rgb(self, imgs: List[Union["PIL.Image.Image", torch.Tensor, np.ndarray]]) -> List[np.ndarray]:
        out = []
        for im in imgs:
            if hasattr(im, "mode"):  # PIL
                im = np.array(im.convert("RGB"))
            elif torch.is_tensor(im):
                if im.ndim == 3 and im.shape[0] in (1, 3):
                    im = im.permute(1, 2, 0).detach().cpu().numpy()
                elif im.ndim == 3 and im.shape[-1] in (1, 3):
                    im = im.detach().cpu().numpy()
                else:
                    raise ValueError("Unsupported tensor image shape")
                if im.dtype != np.uint8:
                    im = (np.clip(im, 0, 1) * 255.0).astype(np.uint8)
            elif isinstance(im, np.ndarray):
                if im.dtype != np.uint8:
                    im = im.astype(np.uint8)
            else:
                raise ValueError("Unsupported image type")
            out.append(im)
        return out

    def _global_pool(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, C, H, W) -> (B, C)
        return F.adaptive_avg_pool2d(feats, 1).flatten(1)

    def _masked_pool(self, feats: torch.Tensor, mask_lowres: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, C, Hf, Wf)
        mask_lowres: (B, 1, 256, 256) — SAM decoder resolution
        We resize mask to feats spatial size and compute masked average pooling.
        """
        B, C, Hf, Wf = feats.shape
        mask = F.interpolate(mask_lowres, size=(Hf, Wf), mode='nearest')  # (B,1,Hf,Wf)
        weight = mask.clamp(min=0.0, max=1.0)
        denom = weight.sum(dim=(2, 3)).clamp(min=1e-6)  # (B,1)
        pooled = (feats * weight).sum(dim=(2, 3)) / denom  # (B,C)
        return pooled

    def forward(
        self,
        images: List[Union["PIL.Image.Image", torch.Tensor, np.ndarray]],
        boxes_xyxy: Optional[torch.Tensor] = None,  # (B,4) in original pixel coords
        gt_lowres_mask: Optional[torch.Tensor] = None,  # (B,1,256,256) optional, for mask-guided pooling
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        If boxes are provided, also returns seg_logits (B,1,256,256).
        For mask-guided pooling:
          - preference order = gt_lowres_mask (if provided) else predicted seg_logits (if available).
        """
        if device is None:
            device = next(self.parameters()).device

        np_imgs = self._ensure_numpy_rgb(images)
        x, orig_sizes = self._preprocess_batch_images(np_imgs)
        x = x.to(device)

        # Encode images
        feats = self.sam.image_encoder(x)  # (B,256,H',W')

        seg_logits = None
        if boxes_xyxy is not None:
            assert boxes_xyxy.shape[0] == feats.shape[0], "boxes batch size must match images"
            boxes_list = [boxes_xyxy[i].detach().cpu().numpy()[None, :] for i in range(boxes_xyxy.shape[0])]
            boxes_t = []
            for b, (h, w) in zip(boxes_list, orig_sizes):
                bt = torch.as_tensor(b, dtype=torch.float32, device=device)
                bt = self.transform.apply_boxes_torch(bt, (h, w))
                boxes_t.append(bt)
            boxes_t = torch.cat(boxes_t, dim=0)
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes_t, masks=None)
            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=feats,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )  # (B,1,256,256)
            seg_logits = low_res_masks

        # Classification feature(s)
        g = self._global_pool(feats)  # (B,256)
        if self.use_mask_guided_pool:
            roi_feat = None
            candidate = gt_lowres_mask if gt_lowres_mask is not None else seg_logits
            if candidate is not None:
                roi_feat = self._masked_pool(feats, candidate)
            if roi_feat is None:
                roi_feat = torch.zeros_like(g)
            cls_feat = torch.cat([g, roi_feat], dim=1)
        else:
            cls_feat = g

        cls_logits = self.cls_head(cls_feat)  # (B,2)
        return cls_logits, seg_logits