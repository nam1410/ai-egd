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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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
    # More aggressive augmentation for small dataset
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # Added
        transforms.RandomRotation(degrees=15),  # Added
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Increased
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Added
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Added
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),  # Added
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
    
    # Get class distribution per patient
    patient_labels = {}
    for i in range(len(ds)):
        _, label, meta = ds[i]
        patient = meta['patient']
        if patient not in patient_labels:
            patient_labels[patient] = []
        patient_labels[patient].append(label.item())
    
    # Try stratified split by patient's majority class
    patient_majority_class = {p: 1 if sum(labels) > len(labels)/2 else 0 
                             for p, labels in patient_labels.items()}
    
    patients_list = list(patients)
    patient_classes = [patient_majority_class[p] for p in patients_list]
    
    try:
        train_p, val_p = train_test_split(
            patients_list, 
            test_size=test_size, 
            random_state=seed, 
            stratify=patient_classes
        )
    except:
        # Fallback to random split if stratification fails
        train_p, val_p = train_test_split(patients_list, test_size=test_size, random_state=seed)
    
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
    weight_per_class = 1.0 / (class_sample_count + 1e-6)
    weights = [weight_per_class[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return images, labels, metadata


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, loader, device, optimizer, criterion, accumulation_steps=2):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    optimizer.zero_grad()
    
    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * images.size(0) * accumulation_steps
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
    all_preds = []
    all_labels = []
    
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--patient_config_json', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vit_small_patch14_dinov2')
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=8)  # Reduced for small dataset
    parser.add_argument('--epochs_stage1', type=int, default=20)  # Reduced
    parser.add_argument('--epochs_stage2', type=int, default=10)  # Reduced
    parser.add_argument('--lr_head', type=float, default=5e-4)  # Reduced
    parser.add_argument('--lr_ft', type=float, default=1e-5)  # Reduced
    parser.add_argument('--unfreeze_blocks', type=int, default=1)  # Reduced
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--weight_decay', type=float, default=0.05)  # Increased
    parser.add_argument('--dropout', type=float, default=0.3)  # For model head
    parser.add_argument('--label_smoothing', type=float, default=0.1)  # Added
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load patient config
    with open(args.patient_config_json, 'r') as f:
        patient_config = json.load(f)

    # Data
    train_tf, val_tf = build_transforms(img_size=args.img_size)
    
    # Create dataset
    full_ds = EndoClassificationDataset(
        data_root=args.data_root,
        patient_config=patient_config,
        transform=train_tf
    )
    
    print(f"Dataset loaded: {len(full_ds)} samples from {len(full_ds.patient_ids())} patients")
    
    # Get train/val split - maybe increase val size for small dataset
    train_idx, val_idx = split_by_patient(full_ds, test_size=0.3, seed=args.seed)  # 30% validation
    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
    
    if len(train_idx) < 50:
        print("WARNING: Very small training set! Consider collecting more data.")
    
    # Create subsets
    train_ds = Subset(full_ds, train_idx)
    val_full_ds = EndoClassificationDataset(
        data_root=args.data_root,
        patient_config=patient_config,
        transform=val_tf
    )
    val_ds = Subset(val_full_ds, val_idx)

    # Weighted sampler
    train_labels = [full_ds[i][1].item() for i in train_idx]
    sampler = make_weighted_sampler(train_labels)
    
    # Print class distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"Train class distribution: {dict(zip(unique, counts))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, 
                              collate_fn=custom_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=custom_collate)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)
    
    # Add dropout to the head if possible
    if hasattr(model, 'head'):
        # Modify the head to include dropout
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features, 2)
        )
        model.head.to(device)

    # Stage 1: train only head with label smoothing
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr_head, weight_decay=args.weight_decay)
    
    # Label smoothing loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10)

    print('Stage 1: training classifier head with frozen backbone...')
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs_stage1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, device, criterion)
        
        scheduler.step(va_loss)
        early_stopping(va_loss)
        
        print(f'[S1][{epoch}/{args.epochs_stage1}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
              f'val_loss={va_loss:.4f} acc={va_acc:.3f} lr={optimizer.param_groups[0]["lr"]:.2e}')
        
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_stage1.pth'))
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print(f'Stage 1 best: epoch {best_epoch}, acc {best_acc:.3f}')
    
    # Load best model from stage 1
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_stage1.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Stage 2: careful fine-tuning
    if best_acc > 0.6:  # Only fine-tune if stage 1 was somewhat successful
        unfreeze_last_blocks(model, n_blocks=args.unfreeze_blocks)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.lr_ft, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
        early_stopping = EarlyStopping(patience=5)
        
        print(f'Stage 2: fine-tuning last {args.unfreeze_blocks} transformer blocks...')
        
        for epoch in range(1, args.epochs_stage2 + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, _acc, _, _ = evaluate(model, val_loader, device, criterion)
            
            scheduler.step()
            early_stopping(va_loss)
            
            print(f'[S2][{epoch}/{args.epochs_stage2}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} '
                  f'val_loss={va_loss:.4f} acc={va_acc:.3f} lr={optimizer.param_groups[0]["lr"]:.2e}')
            
            if va_acc > best_acc:
                best_acc = va_acc
                best_epoch = args.epochs_stage1 + epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    print(f'Done. Best val acc: {best_acc:.3f} at epoch {best_epoch}')
    
    # Final evaluation with best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth' if best_epoch > args.epochs_stage1 else 'best_stage1.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    _, final_acc, preds, labels = evaluate(model, val_loader, device, criterion)
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Class 0', 'Class 1']))


if __name__ == '__main__':
    main()