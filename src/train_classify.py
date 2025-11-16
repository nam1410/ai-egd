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


def custom_collate(batch):
    """Custom collate function to handle the metadata dict"""
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return images, labels, metadata


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
                        help='Root folder for resolving relative paths in patient config.')
    parser.add_argument('--patient_config_json', type=str, required=True,
                        help='Path to patient config JSON file.')
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    args = parser.parse_args()

    seed_everything(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load patient config
    with open(args.patient_config_json, 'r') as f:
        patient_config = json.load(f)

    # Data
    train_tf, val_tf = build_transforms(img_size=args.img_size)
    
    # Create dataset with train transforms first
    full_ds = EndoClassificationDataset(
        data_root=args.data_root,
        patient_config=patient_config,
        transform=train_tf
    )
    
    print(f"Dataset loaded: {len(full_ds)} samples from {len(full_ds.patient_ids())} patients")
    
    # Get train/val split
    train_idx, val_idx = split_by_patient(full_ds, test_size=0.2, seed=args.seed)
    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
    
    # Create subsets
    train_ds = Subset(full_ds, train_idx)
    
    # For validation, we need to create a new dataset with val transforms
    val_full_ds = EndoClassificationDataset(
        data_root=args.data_root,
        patient_config=patient_config,
        transform=val_tf
    )
    val_ds = Subset(val_full_ds, val_idx)

    # Weighted sampler to mitigate class imbalance
    train_labels = [full_ds[i][1].item() for i in train_idx]
    sampler = make_weighted_sampler(train_labels)
    
    # Print class distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"Train class distribution: {dict(zip(unique, counts))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, 
                              collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=custom_collate)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)

    # Stage 1: train only head
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_head, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print('Stage 1: training classifier head with frozen backbone...')
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs_stage1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f'[S1][{epoch}/{args.epochs_stage1}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
              f'val_loss={va_loss:.4f} acc={va_acc:.3f}')
        
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_stage1.pth'))

    print(f'Stage 1 best: epoch {best_epoch}, acc {best_acc:.3f}')

    # Stage 2: unfreeze last N blocks (low LR fine-tune)
    unfreeze_last_blocks(model, n_blocks=args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_ft, weight_decay=0.01)
    
    print(f'Stage 2: fine-tuning last {args.unfreeze_blocks} transformer blocks at low LR...')
    
    for epoch in range(1, args.epochs_stage2 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f'[S2][{epoch}/{args.epochs_stage2}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
              f'val_loss={va_loss:.4f} acc={va_acc:.3f}')
        
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = args.epochs_stage1 + epoch
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))

    print(f'Done. Best val acc: {best_acc:.3f} at epoch {best_epoch}')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_model.pth'))


if __name__ == '__main__':
    main()