from typing import List, Optional
from dataclasses import dataclass


@dataclass
class StructuredElement:
    """Represents a structured element from document processing"""
    content: str
    element_type: str  # heading, paragraph, table, list, etc.
    level: Optional[int] = None  # For headings
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None  # Bounding box coordinates
