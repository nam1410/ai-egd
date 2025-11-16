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


def _extract_key_parts(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract key parts from filename for matching.
    Returns: (record_id, site_with_number, nb_indicator) or None
    
    Examples:
    "Record 11 NB Body 4 Negative.jpg" -> ("record11", "body4", "nb")
    "Record 11 NB Body 4.xml" -> ("record11", "body4", "nb")
    """
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    
    # Check if it has NB (we only want NB images)
    if 'nb' not in name:
        return None
    
    # Extract record number
    record_match = re.search(r'record\s*(\d+)', name)
    if not record_match:
        return None
    record_id = f"record{record_match.group(1)}"
    
    # Extract site (body/antrum) with number
    site_match = re.search(r'(body|antrum)\s*(\d+)?', name)
    if not site_match:
        return None
    
    site = site_match.group(1)
    number = site_match.group(2) if site_match.group(2) else ""
    site_with_number = f"{site}{number}"
    
    return (record_id, site_with_number, "nb")


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

            if not images_dir:
                continue

            images_dir = _join_root(self.data_root, images_dir)

            if not os.path.isdir(images_dir):
                continue
                
            # Handle optional annotations directory
            if ann_dir:
                ann_dir = _join_root(self.data_root, ann_dir)
                if not os.path.isdir(ann_dir):
                    ann_dir = None
            else:
                ann_dir = None

            # Collect images and xmls
            all_img_files = _walk_files(images_dir, IMG_EXTS)
            
            # Filter for NB images only
            img_files = []
            for img_path in all_img_files:
                key_parts = _extract_key_parts(img_path)
                if key_parts:  # This already checks for NB
                    img_files.append(img_path)
            
            # Build XML mapping using flexible key matching
            xml_map = {}
            if ann_dir is not None:
                all_xml_files = _walk_files(ann_dir, {'.xml'})
                for xml_path in all_xml_files:
                    key_parts = _extract_key_parts(xml_path)
                    if key_parts:  # Only NB XMLs
                        # Create a composite key for matching
                        key = f"{key_parts[0]}_{key_parts[1]}"
                        xml_map[key] = xml_path

            patient_had_sample = False

            for img_path in img_files:
                site = _infer_site_from_filename(os.path.basename(img_path))
                img_key_parts = _extract_key_parts(img_path)
                
                if not img_key_parts:  # Should not happen as we pre-filtered
                    continue

                # Prefer site-level report when provided
                label: Optional[int] = None
                if site and (site in cfg):
                    report = cfg.get(site)
                    if isinstance(report, str):
                        label = 0 if report.strip().lower() == 'negative' else 1

                # Find matching XML using flexible key
                xml_path = None
                img_key = f"{img_key_parts[0]}_{img_key_parts[1]}"
                if img_key in xml_map:
                    xml_path = xml_map[img_key]

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
        
        # Return metadata as dict
        metadata = {
            "patient": patient,
            "xml": xml_path,
            "path": img_path,
        }
        
        return im, torch.tensor(label, dtype=torch.long), metadata

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
    
    # Custom collate function to handle dict metadata
    def custom_collate(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        # Don't stack metadata dicts, keep as list
        metadata = [item[2] for item in batch]
        return images, labels, metadata
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate)
    try:
        batch = next(iter(loader))
        images, labels, metas = batch
        print(f"  Batch shapes: images={images.shape}, labels={labels.shape}")
        print(f"  Batch dtypes: images={images.dtype}, labels={labels.dtype}")
        print(f"  Metadata in batch: {len(metas)} items")
    except Exception as e:
        print(f"  ERROR in DataLoader: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Check matching between images and XMLs
    print("\nChecking image-XML matching:")
    matched_count = 0
    unmatched_count = 0
    for i in range(min(10, len(dataset))):
        _, _, meta = dataset[i]
        if meta['xml']:
            matched_count += 1
            img_parts = _extract_key_parts(meta['path'])
            xml_parts = _extract_key_parts(meta['xml'])
            print(f"  Image: {os.path.basename(meta['path'])}")
            print(f"    -> XML: {os.path.basename(meta['xml'])}")
            print(f"    -> Image parts: {img_parts}")
            print(f"    -> XML parts: {xml_parts}")
        else:
            unmatched_count += 1
    print(f"Total checked: {min(10, len(dataset))}, Matched: {matched_count}, Unmatched: {unmatched_count}")