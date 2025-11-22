import argparse, json, os, random, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import GroupKFold
from typing import Dict, List, Tuple

from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.models.dinov2_classifier import create_dinov2_classifier, freeze_backbone


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labs = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return imgs, labs, metas


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss=0; tot=0; correct=0
    for imgs,labs,_ in loader:
        imgs,labs = imgs.to(device), labs.to(device)
        logits = model(imgs)
        loss = criterion(logits,labs)
        tot_loss += loss.item()*imgs.size(0)
        correct += (logits.argmax(1)==labs).sum().item()
        tot += imgs.size(0)
    return tot_loss/max(tot,1), correct/max(tot,1)


def train_epoch(model, loader, device, optimizer, criterion):
    model.train()
    tot_loss=0; tot=0; correct=0
    for imgs,labs,_ in loader:
        imgs,labs = imgs.to(device), labs.to(device)
        logits = model(imgs)
        loss = criterion(logits,labs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()*imgs.size(0)
        correct += (logits.argmax(1)==labs).sum().item()
        tot += imgs.size(0)
    return tot_loss/max(tot,1), correct/max(tot,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--patient_config_json', required=True)
    ap.add_argument('--save_dir', default='./cv_checkpoints')
    ap.add_argument('--splits_file', default='./kfold_splits.npz')
    ap.add_argument('--model_name', default='vit_small_patch14_dinov2')
    ap.add_argument('--img_size', type=int, default=518)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold_index', type=int, required=True, help='0-based index of fold to train')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true', help='Force retrain even if best checkpoint exists')
    args = ap.parse_args()

    assert 0 <= args.fold_index < args.folds, "fold_index out of range"

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.patient_config_json) as f:
        cfg=json.load(f)

    train_tf, val_tf = build_transforms(args.img_size)
    ds_full = EndoClassificationDataset(args.data_root, cfg, transform=train_tf)

    patients = np.array([ds_full[i][2]['patient'] for i in range(len(ds_full))])
    labels = np.array([ds_full[i][1].item() for i in range(len(ds_full))])

    # Generate or load splits
    if os.path.exists(args.splits_file):
        data = np.load(args.splits_file, allow_pickle=True)
        folds_indices = data['folds_indices']
        if len(folds_indices) != args.folds:
            raise ValueError("Stored folds count mismatch.")
    else:
        gkf = GroupKFold(n_splits=args.folds)
        folds_indices = []
        for tr, va in gkf.split(np.zeros(len(ds_full)), labels, groups=patients):
            folds_indices.append({'train': tr, 'val': va})
        np.savez(args.splits_file, folds_indices=folds_indices)
        print(f"Saved splits to {args.splits_file}")

    fold_data = folds_indices[args.fold_index]
    tr_idx = fold_data['train']
    va_idx = fold_data['val']

    # Build val dataset with val transforms
    ds_val_full = EndoClassificationDataset(args.data_root, cfg, transform=val_tf)
    ds_train = Subset(ds_full, tr_idx)
    ds_val = Subset(ds_val_full, va_idx)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_ckpt_path = os.path.join(args.save_dir, f'fold{args.fold_index+1}_best.pth')
    if os.path.exists(best_ckpt_path) and not args.force:
        print(f"Fold {args.fold_index+1} best checkpoint already exists. Use --force to retrain.")
        return

    model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)

    best_acc=0.0
    last_ckpt_path = os.path.join(args.save_dir, f'fold{args.fold_index+1}_last.pth')

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f"[Fold {args.fold_index+1} Ep {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

        torch.save({'epoch': epoch, 'fold': args.fold_index, 'state_dict': model.state_dict(),
                    'val_acc': va_acc}, last_ckpt_path)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'epoch': epoch, 'fold': args.fold_index, 'state_dict': model.state_dict(),
                        'val_acc': va_acc}, best_ckpt_path)

    print(f"Fold {args.fold_index+1} done. Best val acc={best_acc:.3f}")


if __name__ == '__main__':
    main()