import argparse, json, os, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.models.dinov2_classifier import create_dinov2_classifier, freeze_backbone


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labs = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return imgs, labs, metas


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits=[]; all_labels=[]
    for imgs, labs, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labs)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--patient_config_json', required=True)
    ap.add_argument('--splits_file', default='./kfold_splits.npz')
    ap.add_argument('--save_dir', default='./cv_checkpoints')
    ap.add_argument('--model_name', default='vit_small_patch14_dinov2')
    ap.add_argument('--img_size', type=int, default=518)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    with open(args.patient_config_json) as f:
        cfg = json.load(f)

    val_tf = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds_val_full = EndoClassificationDataset(args.data_root, cfg, transform=val_tf)

    if not os.path.exists(args.splits_file):
        raise FileNotFoundError("Splits file not found. Run single-fold trainer first to create it.")

    data = np.load(args.splits_file, allow_pickle=True)
    folds_indices = data['folds_indices']
    assert len(folds_indices) == args.folds, "Folds mismatch."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_fold_metrics = []
    all_labels_total = []
    all_probs_total = []

    for fidx in range(args.folds):
        ckpt_path = os.path.join(args.save_dir, f'fold{fidx+1}_best.pth')
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Missing checkpoint for fold {fidx+1}. Skipping.")
            continue

        print(f"Evaluating fold {fidx+1} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        va_idx = folds_indices[fidx]['val']

        model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)
        freeze_backbone(model)
        model.load_state_dict(ckpt['state_dict'], strict=False)

        va_subset = Subset(ds_val_full, va_idx)
        loader = DataLoader(va_subset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
        logits, labels = predict(model, loader, device)
        probs = torch.softmax(logits, dim=1)[:,1]

        preds = probs.ge(0.5).long()
        acc = (preds == labels).float().mean().item()
        try:
            auc = roc_auc_score(labels.numpy(), probs.numpy())
        except:
            auc = float('nan')

        cm = confusion_matrix(labels.numpy(), preds.numpy())
        report = classification_report(labels.numpy(), preds.numpy(), digits=3)

        print(f"Fold {fidx+1} Acc={acc:.3f} AUC={auc:.3f}")
        print("Confusion matrix:\n", cm)
        print(report)

        all_fold_metrics.append(acc)
        all_labels_total.append(labels.numpy())
        all_probs_total.append(probs.numpy())

    if all_fold_metrics:
        all_labels_concat = np.concatenate(all_labels_total)
        all_probs_concat = np.concatenate(all_probs_total)
        all_preds_concat = (all_probs_concat >= 0.5).astype(int)
        overall_acc = (all_preds_concat == all_labels_concat).mean()
        try:
            overall_auc = roc_auc_score(all_labels_concat, all_probs_concat)
        except:
            overall_auc = float('nan')
        cm_total = confusion_matrix(all_labels_concat, all_preds_concat)
        report_total = classification_report(all_labels_concat, all_preds_concat, digits=3)

        print("\n=== Aggregate (concatenated validation folds) ===")
        print(f"Mean Acc across folds: {np.mean(all_fold_metrics):.3f} ± {np.std(all_fold_metrics):.3f}")
        print(f"Overall Acc (concatenated): {overall_acc:.3f}  Overall AUC: {overall_auc:.3f}")
        print("Confusion matrix:\n", cm_total)
        print(report_total)
    else:
        print("No folds evaluated.")


if __name__ == '__main__':
    main()