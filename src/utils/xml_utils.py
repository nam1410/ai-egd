import xml.etree.ElementTree as ET
from typing import Optional, List


def parse_object_names_from_xml(xml_path: str) -> Optional[List[str]]:
    """
    Returns a list of object 'name' entries from the given VOC-style XML.
    If parsing fails or file doesn't exist, returns None.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        names = []
        for obj in root.findall('object'):
            name_el = obj.find('name')
            if name_el is not None and name_el.text is not None:
                names.append(name_el.text.strip())
        return names
    except Exception:
        return None


def xml_to_binary_label(xml_path: str) -> Optional[int]:
    """
    Fallback rule to assign a binary label from the XML:
      - If any object name is not 'Normal' -> positive (1)
      - Else -> negative (0)
    Returns None if XML can't be parsed.
    """
    names = parse_object_names_from_xml(xml_path)
    if names is None:
        return None
    for n in names:
        if n.lower() != 'normal':
            return 1
    return 0