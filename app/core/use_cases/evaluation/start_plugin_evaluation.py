import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from ...domain.entities.evaluation import EvaluationRun, TestCase
from ...ports.evaluation_repository import EvaluationRepository
from ...ports.repositories import CollectionRepository
from ..context_retrieval import ContextRetrievalUseCase

logger = logging.getLogger(__name__)


class StartPluginEvaluationUseCase:
    """
    Start a plugin-based evaluation.

    Creates an evaluation run and retrieves context for all test questions.
    Returns evaluation_run_id and test data (questions + context) for plugin to generate answers.
    """

    def __init__(
        self,
        evaluation_repo: EvaluationRepository,
        collection_repo: CollectionRepository,
        context_retrieval: ContextRetrievalUseCase,
        test_collection_id: str,
        test_cases_path: str
    ):
        self._evaluation_repo = evaluation_repo
        self._collection_repo = collection_repo
        self._context_retrieval = context_retrieval
        self._test_collection_id = test_collection_id
        self._test_cases_path = test_cases_path

    async def execute(self) -> Dict[str, Any]:
        """
        Start plugin evaluation.

        Returns:
            Dict with:
                - evaluation_run_id: str
                - test_data: List[Dict] with test_case_id, question, retrieved_context
        """
        logger.info("Starting plugin evaluation")

        # Verify test collection exists
        collection = await self._collection_repo.find_by_id(self._test_collection_id)
        if not collection:
            raise ValueError(f"Evaluation test collection not found: {self._test_collection_id}")

        # Load test cases
        test_cases = self._load_test_cases()
        logger.info(f"Loaded {len(test_cases)} test cases")

        # Create evaluation run
        config_snapshot = {
            "collection_id": self._test_collection_id,
            "test_cases_path": self._test_cases_path,
            "evaluation_type": "plugin"
        }

        evaluation_run = EvaluationRun.create(
            collection_id=self._test_collection_id,
            total_questions=len(test_cases),
            config_snapshot=config_snapshot
        )

        await self._evaluation_repo.save_run(evaluation_run)
        logger.info(f"Created evaluation run: {evaluation_run.id}")

        # Retrieve context for each test case
        test_data = []
        for test_case in test_cases:
            logger.debug(f"Retrieving context for test case: {test_case.id}")

            context = await self._context_retrieval.retrieve_context(
                query=test_case.question,
                collection_id=self._test_collection_id,
                top_k=5,
                use_hybrid=True
            )

            test_data.append({
                "test_case_id": test_case.id,
                "question": test_case.question,
                "category": test_case.category,
                "retrieved_context": context,
                "ground_truth": test_case.ground_truth
            })

        logger.info(f"Retrieved context for {len(test_data)} test cases")

        return {
            "evaluation_run_id": evaluation_run.id,
            "test_data": test_data
        }

    def _load_test_cases(self) -> List[TestCase]:
        """Load test cases from JSON file"""
        test_cases_file = Path(self._test_cases_path)

        if not test_cases_file.exists():
            raise FileNotFoundError(f"Test cases file not found: {self._test_cases_path}")

        with open(test_cases_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [TestCase.from_dict(tc) for tc in data.get("test_cases", [])]
