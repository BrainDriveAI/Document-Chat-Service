from abc import ABC, abstractmethod
from typing import Optional, List

from ..domain.entities.evaluation import EvaluationRun, EvaluationResult, TestCase


class EvaluationRepository(ABC):
    """Port interface for evaluation persistence"""

    @abstractmethod
    async def save_run(self, evaluation_run: EvaluationRun) -> EvaluationRun:
        """Save or update an evaluation run"""
        pass

    @abstractmethod
    async def save_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        """Save an evaluation result"""
        pass

    @abstractmethod
    async def find_run_by_id(self, run_id: str) -> Optional[EvaluationRun]:
        """Find evaluation run by ID"""
        pass

    @abstractmethod
    async def find_results_by_run_id(self, run_id: str) -> List[EvaluationResult]:
        """Find all results for an evaluation run"""
        pass

    @abstractmethod
    async def list_runs(self, limit: int = 50) -> List[EvaluationRun]:
        """List recent evaluation runs"""
        pass

    @abstractmethod
    async def get_evaluated_test_case_ids(self, run_id: str) -> List[str]:
        """Get list of test_case_ids that have already been evaluated for a run"""
        pass

    @abstractmethod
    async def save_test_cases(self, evaluation_run_id: str, test_cases: List[TestCase]) -> None:
        """Save test cases for an evaluation run"""
        pass

    @abstractmethod
    async def find_test_cases_by_run_id(self, run_id: str) -> List[TestCase]:
        """Find all test cases for an evaluation run"""
        pass
