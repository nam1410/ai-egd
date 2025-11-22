import argparse, json, os, random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.datasets.endo_multitask_dataset import EndoMultiTaskDataset
from src.models.medsam_with_classifier import MedSamWithClassifier
from src.utils.seg_losses import DiceLoss, resize_mask_like_sam


def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_by_patient(ds, test_size=0.2, seed=42):
    patients = ds.patient_ids()
    tr_p, va_p = train_test_split(patients, test_size=test_size, random_state=seed, shuffle=True)
    tr_idx, va_idx = [], []
    for i in range(len(ds)):
        _, _, meta = ds[i]
        (tr_idx if meta['patient'] in tr_p else va_idx).append(i)
    return tr_idx, va_idx


def weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.array([np.sum(np.array(labels) == t) for t in np.unique(labels)])
    w_pc = 1.0 / (counts + 1e-6)
    weights = [w_pc[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def collate_list_images(batch):
    images = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, labels, metas


@torch.no_grad()
def evaluate_cls(model, loader, device, criterion):
    model.eval()
    total_loss=0.0; total=0; correct=0
    for images, labels, _ in loader:
        labels = labels.to(device)
        logits, _ = model(images, device=device)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss/max(total,1), correct/max(total,1)


@torch.no_grad()
def evaluate_clsseg(model, loader, device, ce_loss, bce_loss, dice_loss, pos_only=True):
    model.eval()
    total_ce=0.0; total_seg=0.0; total=0; correct=0; dice_sum=0.0; dice_n=0
    for images, labels, metas in loader:
        labels = labels.to(device)
        # full-batch classification
        logits, _ = model(images, device=device)
        total_ce += ce_loss(logits, labels).item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # sub-batch for seg (only where masks exist)
        idxs = [i for i,m in enumerate(metas) if m.get("gt_mask") is not None and m.get("box") is not None]
        if len(idxs) == 0:
            continue
        sub_images = [images[i] for i in idxs]
        sub_boxes = torch.stack([metas[i]["box"] for i in idxs]).to(device)
        # prepare lowres GT
        gt_list = []
        for i in idxs:
            gt = metas[i]["gt_mask"]
            H,W = metas[i]["orig_hw"]
            gt_low = resize_mask_like_sam(gt, (H,W))
            gt_list.append(gt_low)
        gt = torch.stack(gt_list).to(device)  # (N,1,256,256)

        _, seg_logits = model(sub_images, boxes_xyxy=sub_boxes, device=device)
        b = bce_loss(seg_logits, gt)
        d = dice_loss(seg_logits, gt)
        total_seg += (b.item() + d.item()) * len(idxs)
        dice_sum += (1.0 - d.item()) * len(idxs)
        dice_n += len(idxs)

    acc = correct/max(total,1)
    mean_dice = dice_sum/max(dice_n,1) if dice_n>0 else 0.0
    return (total_ce/max(total,1)), (total_seg/max(total,1)), acc, mean_dice


def train_epoch_cls(model, loader, device, optimizer, criterion):
    model.train()
    total_loss=0.0; total=0; correct=0
    for images, labels, _ in loader:
        labels = labels.to(device)
        logits, _ = model(images, device=device)
        loss = criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss/max(total,1), correct/max(total,1)


def train_epoch_clsseg(model, loader, device, optimizer, ce_loss, bce_loss, dice_loss, alpha=1.0, beta=0.5, use_mask_guided_pool=False):
    model.train()
    ce_sum=0.0; seg_sum=0.0; total=0; correct=0
    for images, labels, metas in loader:
        labels = labels.to(device)
        optimizer.zero_grad()

        # Build GT masks subset
        idxs = [i for i,m in enumerate(metas) if m.get("gt_mask") is not None and m.get("box") is not None]
        gt_low = None; boxes_t = None; sub_images = None
        if len(idxs) > 0:
            sub_images = [images[i] for i in idxs]
            boxes_t = torch.stack([metas[i]["box"] for i in idxs]).to(device)
            gt_low = torch.stack([
                resize_mask_like_sam(metas[i]["gt_mask"], metas[i]["orig_hw"])
                for i in idxs
            ]).to(device)  # (N,1,256,256)

        # Classification on full batch; if mask-guided pooling, pass GT lowres for imgs that have it
        gt_low_full = None
        if use_mask_guided_pool and len(idxs) > 0:
            # Build a full-batch tensor with zeros except where GT exists
            B = len(images)
            gt_low_full = torch.zeros((B,1,256,256), dtype=torch.float32, device=device)
            for j,i in enumerate(idxs):
                gt_low_full[i] = gt_low[j]
        logits, _ = model(images, gt_lowres_mask=gt_low_full, device=device)
        ce = ce_loss(logits, labels)
        loss = alpha * ce

        # Segmentation forward on the subset with masks
        if gt_low is not None:
            _, seg_logits = model(sub_images, boxes_xyxy=boxes_t, device=device)
            b = bce_loss(seg_logits, gt_low)
            d = dice_loss(seg_logits, gt_low)
            seg_term = b + d
            loss = loss + beta * seg_term
            seg_sum += seg_term.item() * len(idxs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ce_sum += ce.item() * labels.size(0)
        total += labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

    return (ce_sum/max(total,1)), (seg_sum/max(total,1)), correct/max(total,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['cls','clsseg'], required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--patient_config_json', required=True)
    ap.add_argument('--sam_checkpoint', required=True)
    ap.add_argument('--model_type', default='vit_b', choices=['vit_b','vit_l','vit_h'])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alpha', type=float, default=1.0, help='CE weight')
    ap.add_argument('--beta', type=float, default=0.5, help='BCE+Dice weight for seg')
    ap.add_argument('--use_mask_guided_pool', action='store_true')
    ap.add_argument('--save_dir', default='checkpoints_medsam')
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.patient_config_json) as f:
        cfg: Dict = json.load(f)

    # Dataset
    if args.task == 'cls':
        ds_full = EndoClassificationDataset(args.data_root, cfg, transform=None)  # pass PIL to model
    else:
        ds_full = EndoMultiTaskDataset(args.data_root, cfg, transform=None)

    tr_idx, va_idx = split_by_patient(ds_full, test_size=0.2, seed=args.seed)
    ds_tr = Subset(ds_full, tr_idx)

    if args.task == 'cls':
        ds_val_full = EndoClassificationDataset(args.data_root, cfg, transform=None)
    else:
        ds_val_full = EndoMultiTaskDataset(args.data_root, cfg, transform=None)
    ds_va = Subset(ds_val_full, va_idx)

    train_labels = [ds_full[i][1].item() for i in tr_idx]
    sampler = weighted_sampler(train_labels)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate_list_images)
    val_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_list_images)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MedSamWithClassifier(
        checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        num_classes=2,
        freeze_image_encoder=True,
        use_mask_guided_pool=args.use_mask_guided_pool,
    ).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    best_metric = 0.0
    best_path = os.path.join(args.save_dir, f"best_{args.task}.pth")

    for epoch in range(1, args.epochs + 1):
        if args.task == 'cls':
            tr_loss, tr_acc = train_epoch_cls(model, train_loader, device, optimizer, ce_loss)
            va_loss, va_acc = evaluate_cls(model, val_loader, device, ce_loss)
            metric = va_acc
            print(f"[{epoch}/{args.epochs}] CLS: train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")
        else:
            tr_ce, tr_seg, tr_acc = train_epoch_clsseg(model, train_loader, device, optimizer, ce_loss, bce_loss, dice_loss,
                                                       alpha=args.alpha, beta=args.beta, use_mask_guided_pool=args.use_mask_guided_pool)
            va_ce, va_seg, va_acc, va_dice = evaluate_clsseg(model, val_loader, device, ce_loss, bce_loss, dice_loss)
            metric = (va_acc + va_dice) / 2.0
            print(f"[{epoch}/{args.epochs}] CLSSEG: tr_ce={tr_ce:.4f} tr_seg={tr_seg:.4f} tr_acc={tr_acc:.3f} | "
                  f"val_ce={va_ce:.4f} val_seg={va_seg:.4f} val_acc={va_acc:.3f} val_dice={va_dice:.3f}")

        # Save last
        last_path = os.path.join(args.save_dir, f"last_{args.task}.pth")
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metric": metric, "args": vars(args)}, last_path)

        # Save best
        if metric > best_metric:
            best_metric = metric
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metric": metric, "args": vars(args)}, best_path)
            print(f"Saved new best to {best_path} (metric={best_metric:.3f})")

    print(f"Done. Best {args.task} metric={best_metric:.3f}")