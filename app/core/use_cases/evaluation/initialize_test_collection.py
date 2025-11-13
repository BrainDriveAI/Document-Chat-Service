import json
import logging
import os
from pathlib import Path
from typing import List

from ...domain.entities.collection import Collection
from ...domain.entities.document import Document, DocumentType
from ...domain.entities.evaluation import TestCase
from ...domain.exceptions import EvaluationInitializationError, TestCaseLoadError
from ...ports.repositories import CollectionRepository, DocumentRepository
from ..document_management import DocumentManagementUseCase as DocumentManagementUseCase

logger = logging.getLogger(__name__)


class InitializeTestCollectionUseCase:
    """Use case for initializing evaluation test collection on first startup"""

    def __init__(
        self,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        document_processing_use_case: DocumentManagementUseCase,
        test_collection_id: str,
        test_collection_name: str,
        test_docs_dir: str
    ):
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.document_processing_use_case = document_processing_use_case
        self.test_collection_id = test_collection_id
        self.test_collection_name = test_collection_name
        self.test_docs_dir = test_docs_dir

    async def initialize_if_needed(self) -> bool:
        """
        Check if evaluation test collection exists and initialize if needed.

        Returns:
            True if initialization was performed, False if collection already exists
        """
        try:
            # Check if collection exists using fixed ID
            existing_collection = await self.collection_repo.find_by_id(self.test_collection_id)

            if existing_collection:
                logger.info(f"Evaluation test collection '{self.test_collection_name}' already exists")
                return False

            logger.info(f"Initializing evaluation test collection '{self.test_collection_name}'")

            # Create the evaluation collection with fixed ID
            from datetime import datetime, UTC
            now = datetime.now(UTC)
            collection = Collection(
                id=self.test_collection_id,
                name=self.test_collection_name,
                description="Evaluation test collection with sample documents for testing RAG system accuracy",
                color="#FF6B6B",
                created_at=now,
                updated_at=now,
                document_count=0
            )
            await self.collection_repo.save(collection)
            logger.info(f"Created evaluation collection: {collection.id}")

            # Process test documents
            test_docs_path = Path(self.test_docs_dir)
            if not test_docs_path.exists():
                raise EvaluationInitializationError(
                    f"Test docs directory not found: {self.test_docs_dir}"
                )

            # Find all PDF files in the test docs directory
            pdf_files = list(test_docs_path.glob("*.pdf"))
            if not pdf_files:
                raise EvaluationInitializationError(
                    f"No PDF files found in {self.test_docs_dir}"
                )

            logger.info(f"Found {len(pdf_files)} PDF files to process")

            # Process each PDF
            for pdf_file in pdf_files:
                logger.info(f"Processing test document: {pdf_file.name}")

                # Create document entity
                document = Document.create(
                    filename=pdf_file.name,
                    original_filename=pdf_file.name,
                    file_path=str(pdf_file.absolute()),
                    file_size=pdf_file.stat().st_size,
                    document_type=DocumentType.PDF,
                    collection_id=collection.id,
                    metadata={"source": "evaluation_test_data"}
                )

                # Save document
                await self.document_repo.save(document)

                # Process document (extract, chunk, embed, index)
                try:
                    processed_doc = await self.document_processing_use_case.process_document(document)
                    logger.info(f"Successfully processed {pdf_file.name}: {processed_doc.chunk_count} chunks")
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                    # Continue with other documents even if one fails
                    continue

            # Update collection document count
            documents = await self.document_repo.find_by_collection_id(collection.id)
            collection.document_count = len(documents)
            await self.collection_repo.save(collection)

            logger.info(f"Evaluation test collection initialized with {len(documents)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize evaluation test collection: {str(e)}")
            raise EvaluationInitializationError(f"Initialization failed: {str(e)}")

    async def load_test_cases(self) -> List[TestCase]:
        """
        Load test cases from test_cases.json file.

        Returns:
            List of TestCase entities

        Raises:
            TestCaseLoadError: If loading fails
        """
        try:
            test_cases_file = Path(self.test_docs_dir) / "test_cases.json"

            if not test_cases_file.exists():
                raise TestCaseLoadError(f"test_cases.json not found in {self.test_docs_dir}")

            logger.info(f"Loading test cases from {test_cases_file}")

            with open(test_cases_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            test_cases_data = data.get("test_cases", [])
            if not test_cases_data:
                raise TestCaseLoadError("No test cases found in test_cases.json")

            test_cases = [TestCase.from_dict(tc) for tc in test_cases_data]

            logger.info(f"Loaded {len(test_cases)} test cases")
            return test_cases

        except json.JSONDecodeError as e:
            raise TestCaseLoadError(f"Invalid JSON in test_cases.json: {str(e)}")
        except Exception as e:
            raise TestCaseLoadError(f"Failed to load test cases: {str(e)}")
