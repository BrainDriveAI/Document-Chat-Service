import spacy
from spacy_layout import spaCyLayout
import logging
from typing import List, Dict, Any, Tuple, Optional
from ...core.ports.document_processor import DocumentProcessor
from ...core.domain.entities.document import Document
from ...core.domain.entities.document_chunk import DocumentChunk
from ...core.domain.exceptions import DocumentProcessingError, InvalidDocumentTypeError
from ...core.ports.token_service import TokenService


class SpacyLayoutProcessor(DocumentProcessor):
    def __init__(
            self,
            token_service: TokenService,
            spacy_model: str = "en_core_web_sm",
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100
    ):
        self.logger = logging.getLogger(__name__)
        self.token_service = token_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        try:
            # Load spaCy model
            self.nlp = spacy.load(spacy_model)
            self.logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            # Fallback to blank model if specific model not available
            self.nlp = spacy.blank("en")
            self.logger.warning(f"Could not load {spacy_model}, using blank English model")

        # Initialize spaCy Layout with minimal configuration
        self.layout = spaCyLayout(self.nlp)
        self.supported_types = ["pdf", "docx", "doc"]

    async def process_document(self, document: Document) -> Tuple[List[DocumentChunk], str]:
        return None