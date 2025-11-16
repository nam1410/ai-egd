import argparse
import json
import os
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.models.dinov2_classifier import create_dinov2_classifier, freeze_backbone, unfreeze_last_blocks


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int = 518):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def split_by_patient(ds: EndoClassificationDataset, test_size=0.2, seed=42) -> Tuple[List[int], List[int]]:
    patients = ds.patient_ids()
    train_p, val_p = train_test_split(patients, test_size=test_size, random_state=seed, shuffle=True)
    train_idx, val_idx = [], []
    for i in range(len(ds)):
        _, _, meta = ds[i]
        if meta['patient'] in train_p:
            train_idx.append(i)
        else:
            val_idx.append(i)
    return train_idx, val_idx


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    class_sample_count = np.array([np.sum(np.array(labels) == t) for t in np.unique(labels)])
    # Inverse frequency
    weight_per_class = 1.0 / (class_sample_count + 1e-6)
    weights = [weight_per_class[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def load_patient_site_map(json_path: str) -> Dict:
    """
    JSON file format example:
    {
      "May 9-1438": {"Antrum": "Chronic Gastritis", "Body": "Chronic Gastritis and H-pylori"},
      "May 9-1528": {"Antrum": "Chronic Gastritis", "Body": "Negative"},
      "May 16-0746": {"Antrum": "Chronic Gastritis and H-pylori", "Body": "Chronic Gastritis and H-pylori"},
      "May 16-0910": {"Antrum": "Negative", "Body": "Negative"},
      "May 16-0952": {"Antrum": "Chronic Gastritis", "Body": "Negative"}
    }
    """
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path, 'r') as f:
        return json.load(f)


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root folder containing patient subfolders with images + XMLs.')
    parser.add_argument('--patient_site_json', type=str, default='',
                        help='Optional JSON mapping patient->site->report. If absent, falls back to XML.')
    parser.add_argument('--model_name', type=str, default='vit_small_patch14_dinov2',
                        help='timm model name for DINOv2 backbone.')
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs_stage1', type=int, default=5, help='Frozen backbone stage.')
    parser.add_argument('--epochs_stage2', type=int, default=10, help='Partial unfreeze stage.')
    parser.add_argument('--lr_head', type=float, default=1e-3)
    parser.add_argument('--lr_ft', type=float, default=5e-5)
    parser.add_argument('--unfreeze_blocks', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Data
    patient_site_map = load_patient_site_map(args.patient_site_json)
    train_tf, val_tf = build_transforms(img_size=args.img_size)
    full_ds = EndoClassificationDataset(
        root_dir=args.data_root,
        patient_site_map=patient_site_map,
        transform=train_tf  # will override for val subset below
    )
    train_idx, val_idx = split_by_patient(full_ds, test_size=0.2, seed=args.seed)
    train_ds = Subset(full_ds, train_idx)
    # Clone dataset with val transforms
    val_ds = Subset(EndoClassificationDataset(
        root_dir=args.data_root,
        patient_site_map=patient_site_map,
        transform=val_tf
    ), val_idx)

    # Weighted sampler to mitigate class imbalance
    train_labels = [full_ds[i][1].item() for i in train_idx]
    sampler = make_weighted_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)

    # Stage 1: train only head
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_head, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print('Stage 1: training classifier head with frozen backbone...')
    best_acc = 0.0
    for epoch in range(1, args.epochs_stage1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f'[S1][{epoch}/{args.epochs_stage1}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
              f'val_loss={va_loss:.4f} acc={va_acc:.3f}')
        best_acc = max(best_acc, va_acc)

    # Stage 2: unfreeze last N blocks (low LR fine-tune)
    unfreeze_last_blocks(model, n_blocks=args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_ft, weight_decay=0.01)
    print('Stage 2: fine-tuning last transformer blocks at low LR...')
    for epoch in range(1, args.epochs_stage2 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f'[S2][{epoch}/{args.epochs_stage2}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
              f'val_loss={va_loss:.4f} acc={va_acc:.3f}')
        best_acc = max(best_acc, va_acc)

    print(f'Done. Best val acc: {best_acc:.3f}')


if __name__ == '__main__':
    main()