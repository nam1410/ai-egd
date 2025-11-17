import argparse, json, os, torch, random, numpy as np
from sklearn.model_selection import GroupKFold
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.models.dinov2_classifier import create_dinov2_classifier, freeze_backbone

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labs = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return imgs,labs,metas

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
    ap.add_argument('--model_name', default='vit_small_patch14_dinov2')
    ap.add_argument('--img_size', type=int, default=518)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', type=str, default='./cv_checkpoints')
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.patient_config_json) as f:
        cfg=json.load(f)

    tf = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = EndoClassificationDataset(args.data_root,cfg,transform=tf)

    # Build groups array: patient per sample
    patients = [ds[i][2]['patient'] for i in range(len(ds))]
    y = np.array([ds[i][1].item() for i in range(len(ds))])
    gkf = GroupKFold(n_splits=args.folds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_metrics=[]
    for fold,(tr_idx,va_idx) in enumerate(gkf.split(np.zeros(len(ds)), y, groups=patients)):
        print(f"\n---- Fold {fold+1}/{args.folds} ----")
        tr_subset = Subset(ds, tr_idx)
        va_subset = Subset(ds, va_idx)
        train_loader = DataLoader(tr_subset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=collate)
        val_loader = DataLoader(va_subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, collate_fn=collate)

        model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)
        freeze_backbone(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.lr, weight_decay=0.05)

        best_acc=0.0
        for epoch in range(1,args.epochs+1):
            tr_loss, tr_acc = train_epoch(model, train_loader, device, optimizer, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)
            print(f"[Fold {fold+1} Ep {epoch}] train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                  f"val_loss={va_loss:.4f} acc={va_acc:.3f}")
            if va_acc>best_acc:
                best_acc=va_acc
                torch.save({"fold":fold,"epoch":epoch,"state_dict":model.state_dict(),"val_acc":va_acc},
                           os.path.join(args.save_dir,f"fold{fold+1}_best.pth"))
        fold_metrics.append(best_acc)

    print(f"\nCV results: {fold_metrics}  Mean={np.mean(fold_metrics):.3f}  Std={np.std(fold_metrics):.3f}")

if __name__=='__main__':
    main()