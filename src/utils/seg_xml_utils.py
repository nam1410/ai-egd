import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


def _parse_polygon_string(s: str) -> List[Tuple[float, float]]:
    pts = []
    for tok in s.strip().split(';'):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(',')
        if len(parts) < 2:
            continue
        x = float(parts[0]); y = float(parts[1])
        pts.append((x, y))
    return pts


def positive_polygons_and_size(xml_path: str) -> Tuple[List[List[Tuple[float, float]]], Tuple[int, int]]:
    """
    Returns:
      - list of polygons (each polygon is list[(x,y), ...]) for objects with name != 'Normal'
      - (width, height) from <size> in XML
    """
    tree = ET.parse(xml_path); root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text); h = int(size.find('height').text)
    polys: List[List[Tuple[float, float]]] = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        name = (name_el.text or '').strip().lower() if name_el is not None else ''
        NEGATIVES = ['normal', 'useless']
        # if name == 'normal':
        if name in NEGATIVES:
            continue
        mask_el = obj.find('mask')
        if mask_el is not None and mask_el.text:
            poly = _parse_polygon_string(mask_el.text)
            if len(poly) >= 3:
                polys.append(poly)
            continue
        # fallback: bbox as polygon
        bb = obj.find('bndbox')
        if bb is not None:
            xmin = float(bb.find('xmin').text); ymin = float(bb.find('ymin').text)
            xmax = float(bb.find('xmax').text); ymax = float(bb.find('ymax').text)
            polys.append([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    return polys, (w, h)


def rasterize_positive_mask(xml_path: str) -> np.ndarray:
    """
    Rasterize union of positive polygons into a binary mask (H, W) uint8 {0,1}.
    """
    polys, (w, h) = positive_polygons_and_size(xml_path)
    mask = Image.new('L', (w, h), 0)
    if polys:
        draw = ImageDraw.Draw(mask)
        for poly in polys:
            draw.polygon(poly, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Return tight xyxy box for non-zero region in mask. None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())