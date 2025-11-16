# Endoscopy Binary Classification (DINOv2)

This is a minimal starter to:
- Train an image-level binary classifier for “tumor” (1) vs “non-tumor” (0)
- Use patient-level splits to avoid leakage
- Start with a DINOv2 backbone and a simple linear head, then fine-tune last blocks at low LR

Later you can extend this to segmentation (see next steps).

## Install

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data layout

Root directory should contain patient folders. Inside each patient folder are images and matching XMLs.

```
/data_root/
  May 9-1438/
    May 9-1438-17-NB-Antrum.jpeg
    May 9-1438-17-NB-Antrum.xml
    May 9-1438-26-NB-Body.jpeg
    May 9-1438-26-NB-Body.xml
  May 9-1528/
    ...
```

- If you have a patient/site pathology table, convert it to JSON:

```json
{
  "May 9-1438": {"Antrum": "Chronic Gastritis", "Body": "Chronic Gastritis and H-pylori"},
  "May 9-1528": {"Antrum": "Chronic Gastritis", "Body": "Negative"},
  "May 16-0746": {"Antrum": "Chronic Gastritis and H-pylori", "Body": "Chronic Gastritis and H-pylori"},
  "May 16-0910": {"Antrum": "Negative", "Body": "Negative"},
  "May 16-0952": {"Antrum": "Chronic Gastritis", "Body": "Negative"}
}
```

Labeling rule:
- "Negative" => 0 (non-tumor)
- anything else => 1 (tumor)

If you don’t provide the JSON, we fall back to XML:
- if any `<object><name>` in the XML != "Normal" => 1
- else => 0

## Train

```
python -m src.train_classify \
  --data_root /path/to/data_root \
  --patient_site_json /path/to/patient_site.json \
  --model_name vit_small_patch14_dinov2 \
  --img_size 518 \
  --batch_size 16 \
  --epochs_stage1 5 \
  --epochs_stage2 10 \
  --lr_head 1e-3 \
  --lr_ft 5e-5 \
  --unfreeze_blocks 2
```

Notes:
- `img_size=518` matches common DINOv2 training crop for ViT patch-14, but you can use 448/392/336 if VRAM is limited.
- We use a weighted sampler to mitigate class imbalance.

## Roadmap to add Segmentation

Once you’re satisfied with classification:

1. Generate binary masks from your XML polygons
   - Positive mask: union of all objects whose `name != "Normal"`
   - Negative mask: background

2. Two practical options:
   - MedSAM fine-tuning:
     - Use your `<bndbox>` as point/box prompts and `<mask>` polygons as supervision.
     - Fine-tune the MedSAM decoder only for binary masks.
   - Single-backbone multi-task:
     - Keep DINOv2 as encoder.
     - Add a lightweight decoder (e.g., FPN/UPerNet-like) for per-pixel binary logits.
     - Train with a combined loss: CE for classification + Dice/BCE for segmentation.

3. Start small:
   - Freeze encoder, train decoder/head only.
   - Then unfreeze last blocks at very low LR (1e-5 to 5e-6).

4. Evaluation:
   - Classification: accuracy, ROC-AUC
   - Segmentation: Dice, IoU (per image and per patient)

If you want, I can extend this scaffold with a binary segmentation dataloader (XML polygon rasterization) and a simple ViT-decoder head, or prepare a MedSAM fine-tuning script wired to your XMLs.