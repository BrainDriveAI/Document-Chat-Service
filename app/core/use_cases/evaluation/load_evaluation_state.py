import logging
from typing import Dict, Any, Optional

from ...ports.evaluation_state_repository import EvaluationStateRepository
from ...ports.evaluation_repository import EvaluationRepository

logger = logging.getLogger(__name__)


class StateNotFoundError(Exception):
    """Raised when evaluation state is not found"""
    pass


class LoadEvaluationStateUseCase:
    """Use case for loading evaluation state"""

    def __init__(
        self,
        state_repository: EvaluationStateRepository,
        evaluation_repository: EvaluationRepository
    ):
        self._state_repo = state_repository
        self._eval_repo = evaluation_repository

    async def execute(
        self,
        evaluation_run_id: str,
        user_id: Optional[str] = None,
        max_age_days: int = 7
    ) -> Dict[str, Any]:
        """
        Load evaluation state for resume.

        Args:
            evaluation_run_id: The evaluation run ID
            user_id: Optional user ID filter
            max_age_days: Maximum age for expiry calculation (default 7 days)

        Returns:
            Dictionary with state data and metadata

        Raises:
            StateNotFoundError: If state not found or user_id mismatch
        """
        # Fetch state by run_id
        state = await self._state_repo.find_by_run_id(evaluation_run_id)

        if not state:
            raise StateNotFoundError(f"Evaluation state not found: {evaluation_run_id}")

        # Filter by user_id if provided (return 404 if mismatch)
        if user_id and state.user_id != user_id:
            raise StateNotFoundError(
                f"Evaluation state not found for user {user_id}: {evaluation_run_id}"
            )

        # Fetch backend evaluation status
        evaluation_run = await self._eval_repo.find_run_by_id(evaluation_run_id)
        backend_status = evaluation_run.status.value if evaluation_run else "unknown"

        # Calculate metadata
        metadata = {
            "age_hours": state.age_hours,
            "age_days": state.age_days,
            "is_expired": state.is_expired(max_age_days),
            "will_expire_in_hours": state.will_expire_in_hours(max_age_days),
            "backend_evaluation_status": backend_status
        }

        logger.info(
            f"Loaded evaluation state for run: {evaluation_run_id}, "
            f"age: {state.age_hours:.2f}h, expired: {metadata['is_expired']}"
        )

        return {
            "state": state.state_data,
            "metadata": metadata
        }
