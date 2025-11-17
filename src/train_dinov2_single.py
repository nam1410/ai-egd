import argparse, json, os, random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.models.dinov2_classifier import create_dinov2_classifier, freeze_backbone

def seed_everything(seed: int = 42):
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

def split_by_patient(ds: EndoClassificationDataset, test_size=0.2, seed=42):
    patients = ds.patient_ids()
    # majority class per patient for stratification
    patient_labels = {p: [] for p in patients}
    for i in range(len(ds)):
        _, lab, meta = ds[i]
        patient_labels[meta['patient']].append(lab.item())
    maj = [1 if (sum(patient_labels[p]) > len(patient_labels[p])/2) else 0 for p in patients]
    try:
        tr_p, va_p = train_test_split(patients, test_size=test_size, random_state=seed, stratify=maj)
    except:
        tr_p, va_p = train_test_split(patients, test_size=test_size, random_state=seed)
    tr_idx, va_idx = [], []
    for i in range(len(ds)):
        _, _, meta = ds[i]
        (tr_idx if meta['patient'] in tr_p else va_idx).append(i)
    return tr_idx, va_idx

def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.array([np.sum(np.array(labels)==t) for t in np.unique(labels)])
    inv = 1.0/(counts+1e-6)
    weights = [inv[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss=0; tot=0; correct=0
    all_logits=[]; all_targets=[]
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        tot_loss += loss.item()*images.size(0)
        preds = logits.argmax(1)
        correct += (preds==labels).sum().item()
        tot += images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())
    return tot_loss/max(tot,1), correct/max(tot,1), torch.cat(all_logits), torch.cat(all_targets)

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    tot_loss=0; tot=0; correct=0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()*images.size(0)
        correct += (logits.argmax(1)==labels).sum().item()
        tot += images.size(0)
    return tot_loss/max(tot,1), correct/max(tot,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--patient_config_json', required=True)
    ap.add_argument('--model_name', default='vit_small_patch14_dinov2')
    ap.add_argument('--img_size', type=int, default=518)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=5e-4)   # typical linear probe LR
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    ap.add_argument('--save_dir', type=str, default='./checkpoints_dinov2_single')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.patient_config_json) as f:
        cfg = json.load(f)

    train_tf, val_tf = build_transforms(args.img_size)
    ds_full = EndoClassificationDataset(args.data_root, cfg, transform=train_tf)
    tr_idx, va_idx = split_by_patient(ds_full, test_size=0.2, seed=args.seed)
    ds_train = Subset(ds_full, tr_idx)
    ds_val_full = EndoClassificationDataset(args.data_root, cfg, transform=val_tf)
    ds_val = Subset(ds_val_full, va_idx)

    train_labels = [ds_full[i][1].item() for i in tr_idx]
    sampler = make_weighted_sampler(train_labels)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        labs = torch.stack([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return imgs, labs, metas

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)
    freeze_backbone(model)  # pure linear probe

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)

    best_acc=0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        va_loss, va_acc, va_logits, va_targets = evaluate(model, val_loader, device, criterion)
        print(f"[{epoch}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc>best_acc:
            best_acc=va_acc
            torch.save({"epoch":epoch,"state_dict":model.state_dict(),"val_acc":va_acc},
                       os.path.join(args.save_dir,"best_linear_probe.pth"))
    print(f"Best val acc: {best_acc:.3f}")

if __name__=='__main__':
    main()