from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.entities.evaluation_state import EvaluationState


class EvaluationStateRepository(ABC):
    """Port interface for evaluation state persistence"""

    @abstractmethod
    async def save_state(self, state: EvaluationState) -> EvaluationState:
        """
        Save or update evaluation state (upsert operation).

        If a state with the same evaluation_run_id exists, update it.
        Otherwise, create a new state record.

        Args:
            state: EvaluationState entity to save

        Returns:
            The saved EvaluationState entity
        """
        pass

    @abstractmethod
    async def find_by_run_id(self, evaluation_run_id: str) -> Optional[EvaluationState]:
        """
        Find evaluation state by run ID.

        Args:
            evaluation_run_id: The evaluation run identifier

        Returns:
            EvaluationState if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_by_run_id(self, evaluation_run_id: str) -> bool:
        """
        Delete evaluation state by run ID (idempotent).

        Args:
            evaluation_run_id: The evaluation run identifier

        Returns:
            True if deleted, False if not found (but still success)
        """
        pass

    @abstractmethod
    async def list_all(
        self,
        user_id: Optional[str] = None,
        include_expired: bool = False,
        max_age_days: int = 7
    ) -> List[EvaluationState]:
        """
        List evaluation states with optional filtering.

        Args:
            user_id: Filter by user ID if provided
            include_expired: If False, exclude states older than max_age_days
            max_age_days: Maximum age threshold for filtering (default 7 days)

        Returns:
            List of EvaluationState entities matching criteria
        """
        pass
