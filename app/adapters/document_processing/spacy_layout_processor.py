import spacy
from spacy_layout import spaCyLayout
import pandas as pd
from typing import List, Dict, Any, Tuple
from ...core.ports.document_processor import DocumentProcessor
from ...core.domain.entities.document import Document
from ...core.domain.entities.document_chunk import DocumentChunk
from ...core.domain.exceptions import DocumentProcessingError, InvalidDocumentTypeError
from ...core.ports.chunking_strategy import ChunkingStrategy
from .chunking_strategies import HierarchicalChunkingStrategy


class SpacyLayoutProcessor(DocumentProcessor):
    """spaCy Layout implementation of document processor"""

    def __init__(
            self,
            spacy_model: str = "en_core_web_sm",
            chunking_strategy: ChunkingStrategy = None
    ):
        try:
            # Load spaCy model
            self.nlp = spacy.load(spacy_model)
        except OSError:
            # Fallback to blank model if specific model not available
            self.nlp = spacy.blank("en")

        # Initialize spaCy Layout
        self.layout = spaCyLayout(self.nlp, display_table=self._format_table)

        # Set chunking strategy
        self.chunking_strategy = chunking_strategy or HierarchicalChunkingStrategy()

        self.supported_types = ["pdf", "docx", "doc"]

    async def process_document(self, document: Document) -> Tuple[List[DocumentChunk], str]:
        """Process a document and return structured chunks and complete text"""
        try:
            if document.document_type.value not in self.supported_types:
                raise InvalidDocumentTypeError(
                    f"Document type {document.document_type.value} not supported"
                )

            # Process document with spaCy Layout
            doc = self.layout(document.file_path)

            # Extract structured content
            structured_content = self._extract_structured_content(doc)

            # Generate chunks using strategy
            doc_chunks = await self.chunking_strategy.create_chunks(
                document=document,
                structured_content=structured_content
            )

            # Get complete text from doc
            complete_text = doc.text

            return doc_chunks, complete_text

        except Exception as e:
            raise DocumentProcessingError(f"Failed to process document {document.filename}: {str(e)}")

    def _extract_structured_content(self, doc) -> Dict[str, Any]:
        """Extract structured content from spaCy Doc"""
        content = {
            "full_text": doc.text,
            "sections": [],
            "tables": [],
            "metadata": {
                "total_pages": len(doc._.layout.pages) if hasattr(doc._, 'layout') else 1,
                "total_sections": len(doc.spans.get("layout", [])),
                "total_tables": len(doc._.tables) if hasattr(doc._, 'tables') else 0
            }
        }

        # Extract sections with their structure
        for span in doc.spans.get("layout", []):
            section_info = {
                "text": span.text,
                "label": span.label_,
                "start_char": int(span.start_char),
                "end_char": int(span.end_char),
                "token_start": int(span.start),
                "token_end": int(span.end),
                "heading": str(getattr(span._, 'heading', '')) if hasattr(span, '_') and getattr(span._, 'heading', None) is not None else '',
                "layout_info": self._extract_layout_info(span)
            }
            content["sections"].append(section_info)

        # Extract tables
        if hasattr(doc, '_') and hasattr(doc._, 'tables'):
            for table in doc._.tables:
                table_info = {
                    "data": str(table._.data) if hasattr(table, '_') and hasattr(table._, 'data') and table._.data is not None else '',
                    "text": table.text,
                    "start_char": int(table.start_char),
                    "end_char": int(table.end_char),
                    "layout_info": self._extract_layout_info(table)
                }
                content["tables"].append(table_info)

        return content

    def _extract_layout_info(self, span) -> Dict[str, Any]:
        """Extract layout information from a span"""
        layout_info = {}

        if hasattr(span, '_') and hasattr(span._, 'layout'):
            layout = span._.layout
            if hasattr(layout, 'bbox'):
                layout_info["bbox"] = {
                    "x0": float(layout.bbox.x0),
                    "y0": float(layout.bbox.y0),
                    "x1": float(layout.bbox.x1),
                    "y1": float(layout.bbox.y1)
                }
            if hasattr(layout, 'page'):
                layout_info["page"] = int(layout.page)

        return layout_info

    def _format_table(self, df: pd.DataFrame) -> str:
        """Format table for display in text"""
        if df.empty:
            return "TABLE (empty)"

        # Create a readable table representation
        headers = ", ".join(df.columns.tolist())
        row_count = len(df)

        return f"TABLE: [{headers}] ({row_count} rows)"

    def get_supported_types(self) -> List[str]:
        """Return list of supported document types"""
        return self.supported_types.copy()
