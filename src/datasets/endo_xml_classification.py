import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.utils.xml_utils import xml_to_binary_label


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _infer_site_from_filename(name: str) -> Optional[str]:
    n = name.lower()
    if 'body' in n:
        return 'Body'
    if 'antrum' in n:
        return 'Antrum'
    return None


def _normalize_stem(path: str) -> str:
    """
    Returns a normalized key for matching image <-> xml:
    - lowercase
    - removes all non-alphanumeric chars
    Example:
      'May 9-1438-26-NB-Body' and 'May_9-1438-26-NB_Body' -> same key
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r'[^a-z0-9]+', '', stem.lower())


def _join_root(root: str, maybe_rel: str) -> str:
    if os.path.isabs(maybe_rel):
        return maybe_rel
    return os.path.join(root, maybe_rel)


def _walk_files(root_dir: str, exts: Optional[set] = None) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if exts is None:
                files.append(os.path.join(dp, fn))
            else:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    files.append(os.path.join(dp, fn))
    return files


class EndoClassificationDataset(Dataset):
    """
    Binary classification dataset built from a patient-scoped JSON that specifies
    where to find images and annotations for each patient.

    JSON schema per patient:
    {
      "May 9-1438": {
        "images": "relative/or/absolute/path/to/images",
        "annotations": "relative/or/absolute/path/to/xmls",
        "Antrum": "Chronic Gastritis",
        "Body": "Chronic Gastritis and H-pylori"
      },
      ...
    }

    Labeling rule:
      - If site-level report is provided (for inferred site 'Body'/'Antrum'):
          "Negative" -> 0 (non-tumor), anything else -> 1 (tumor)
      - Else, fallback to XML for that image:
          any <object><name> != "Normal" -> 1, else 0
    """
    def __init__(
        self,
        data_root: str,
        patient_config: Dict[str, Dict[str, str]],
        transform=None,
    ):
        """
        Args:
            data_root: Root path used to resolve relative paths in patient_config.
            patient_config: dict as described above.
            transform: torchvision transforms for image tensors.
        """
        self.data_root = data_root
        self.patient_config = patient_config
        self.transform = transform

        # Build samples
        # Each sample: (image_path, label, patient_id, xml_path)
        self.samples: List[Tuple[str, int, str, Optional[str]]] = []
        self._patients_with_data: List[str] = []

        for patient, cfg in patient_config.items():
            images_dir = cfg.get('images', '')
            ann_dir = cfg.get('annotations', '')

            if not images_dir or not ann_dir:
                # Skip patients without both paths
                continue

            images_dir = _join_root(self.data_root, images_dir)
            ann_dir = _join_root(self.data_root, ann_dir)

            if not os.path.isdir(images_dir):
                continue
            if not os.path.isdir(ann_dir):
                # We allow missing annotations dir (label may come from site report),
                # but keep ann_dir None to signal no XML.
                ann_dir = None

            # Collect images and xmls
            img_files = _walk_files(images_dir, IMG_EXTS)
            xml_map = {}
            if ann_dir is not None:
                xml_files = _walk_files(ann_dir, {'.xml'})
                xml_map = { _normalize_stem(p): p for p in xml_files }

            patient_had_sample = False

            for img_path in img_files:
                site = _infer_site_from_filename(os.path.basename(img_path))

                # Prefer site-level report when provided
                label: Optional[int] = None
                if site and (site in cfg):
                    report = cfg.get(site)
                    if isinstance(report, str):
                        label = 0 if report.strip().lower() == 'negative' else 1

                # Find best-matching XML for this image (for fallback or metadata)
                xml_path = None
                key = _normalize_stem(img_path)
                if key in xml_map:
                    xml_path = xml_map[key]

                # Fallback to XML-derived label if still unknown
                if label is None and xml_path is not None:
                    label = xml_to_binary_label(xml_path)

                # If we still can't label the image, skip it
                if label is None:
                    continue

                self.samples.append((img_path, int(label), patient, xml_path))
                patient_had_sample = True

            if patient_had_sample:
                self._patients_with_data.append(patient)

        if len(self.samples) == 0:
            raise RuntimeError(
                "No labeled images found. Check patient_config JSON paths and labels."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, patient, xml_path = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, torch.tensor(label, dtype=torch.long), {
            "patient": patient,
            "xml": xml_path,
            "path": img_path,
        }

    def patient_ids(self) -> List[str]:
        # Return only those patients that contributed at least one sample
        return list(sorted(set(self._patients_with_data)))
    

if __name__ == "__main__":
    # Simple test to verify dataset loading
    import json
    from torchvision import transforms
    from collections import Counter
    
    #change these paths as needed
    data_root = "/lustre06/project/6103394/ofarooq/AIEGD_datasets/"
    config_path = "/lustre06/project/6103394/ofarooq/ai-egd/src/datasets/patient_config.json"
    
    with open(config_path, 'r') as f:
        patient_config = json.load(f)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = EndoClassificationDataset(
            data_root=data_root,
            patient_config=patient_config,
            transform=transform,
        )
    except RuntimeError as e:
        print(f"Failed to create dataset: {e}")
        exit(1)
    
    print(f"Loaded {len(dataset)} samples from {len(dataset.patient_ids())} patients.")
    
    # 1. Check class distribution
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(labels)
    print(f"\nClass distribution: {dict(label_counts)}")
    if len(label_counts) < 2:
        print("WARNING: Only one class found in dataset!")
    
    # 2. Check samples per patient
    patient_counts = Counter()
    for i in range(len(dataset)):
        _, _, meta = dataset[i]
        patient_counts[meta['patient']] += 1
    
    print(f"\nSamples per patient (showing first 5):")
    for patient, count in list(patient_counts.most_common())[:5]:
        print(f"  {patient}: {count} samples")
    
    # 3. Check XML coverage
    samples_with_xml = sum(1 for i in range(len(dataset)) if dataset[i][2]['xml'] is not None)
    print(f"\nXML annotation coverage: {samples_with_xml}/{len(dataset)} ({samples_with_xml/len(dataset)*100:.1f}%)")
    
    # 4. Test a few samples in detail
    print("\nDetailed sample inspection (first 5):")
    for i in range(min(5, len(dataset))):
        img, label, meta = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Label: {label.item()}")
        print(f"  Patient: {meta['patient']}")
        print(f"  Image path: {meta['path']}")
        print(f"  XML path: {meta['xml'] or 'None'}")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Check if file actually exists
        if not os.path.exists(meta['path']):
            print(f"  WARNING: Image file does not exist!")
        if meta['xml'] and not os.path.exists(meta['xml']):
            print(f"  WARNING: XML file does not exist!")
    
    # 5. Test DataLoader integration
    print("\nTesting DataLoader integration:")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    try:
        batch = next(iter(loader))
        images, labels, metas = batch
        print(f"  Batch shapes: images={images.shape}, labels={labels.shape}")
        print(f"  Batch dtypes: images={images.dtype}, labels={labels.dtype}")
    except Exception as e:
        print(f"  ERROR in DataLoader: {e}")
    
    # 6. Check for duplicate normalized stems (potential matching issues)
    print("\nChecking for potential filename normalization conflicts:")
    stem_to_paths = {}
    for i in range(len(dataset)):
        _, _, meta = dataset[i]
        stem = _normalize_stem(meta['path'])
        if stem not in stem_to_paths:
            stem_to_paths[stem] = []
        stem_to_paths[stem].append(meta['path'])
    
    conflicts = {k: v for k, v in stem_to_paths.items() if len(v) > 1}
    if conflicts:
        print(f"  Found {len(conflicts)} potential conflicts:")
        for stem, paths in list(conflicts.items())[:3]:
            print(f"    {stem}: {paths}")
    else:
        print("  No normalization conflicts found.")
    
    # 7. Verify site inference
    print("\nTesting site inference:")
    test_names = ["patient-body-1.jpg", "patient-antrum-2.jpg", "patient-other-3.jpg"]
    for name in test_names:
        site = _infer_site_from_filename(name)
        print(f"  {name} -> {site}")