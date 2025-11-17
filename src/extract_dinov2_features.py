import argparse, json, os, torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from src.datasets.endo_xml_classification import EndoClassificationDataset
from src.models.dinov2_classifier import create_dinov2_classifier, freeze_backbone

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--patient_config_json', required=True)
    ap.add_argument('--model_name', default='vit_small_patch14_dinov2')
    ap.add_argument('--img_size', type=int, default=518)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_npz', type=str, default='dinov2_features.npz')
    args = ap.parse_args()

   

    with open(args.patient_config_json) as f:
        cfg = json.load(f)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds = EndoClassificationDataset(args.data_root, cfg, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create model and remove classification head to get backbone features
    model = create_dinov2_classifier(num_classes=2, model_name=args.model_name).to(device)
    freeze_backbone(model)
    model.head = torch.nn.Identity()  # so forward returns features before linear layer

    features=[]; labels=[]; patients=[]
    with torch.no_grad():
        for imgs, labs, metas in loader:
            imgs = imgs.to(device)
            feats = model(imgs)  # shape (B, feature_dim)
            features.append(feats.cpu().numpy())
            labels.append(labs.numpy())
            patients.extend([m['patient'] for m in metas])
    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    np.savez(args.out_npz, X=X, y=y, patients=np.array(patients))
    print(f"Saved features: {X.shape} -> {args.out_npz}")

if __name__=='__main__':
    main()