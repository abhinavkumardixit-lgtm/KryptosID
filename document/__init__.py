"""Document Rectification and Region Extraction Package.

Implements Section 3.5 of DESIGN.md:
- locate.py: Document boundary detection and 4-point perspective transform rectification.
- regions.py: ID photo and MRZ (Machine Readable Zone) bounding box extraction.
"""

from document.locate import detect_and_rectify_document
from document.regions import extract_id_regions

__all__ = [
    "detect_and_rectify_document",
    "extract_id_regions",
]
