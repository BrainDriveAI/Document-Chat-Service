import logging
from typing import Optional

from ...ports.evaluation_state_repository import EvaluationStateRepository

logger = logging.getLogger(__name__)


class DeleteEvaluationStateUseCase:
    """Use case for deleting evaluation state"""

    def __init__(self, state_repository: EvaluationStateRepository):
        self._state_repo = state_repository

    async def execute(
        self,
        evaluation_run_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """
        Delete evaluation state (idempotent).

        Args:
            evaluation_run_id: The evaluation run ID
            user_id: Optional user ID for filtering (currently not enforced, for future use)

        Note:
            This operation is idempotent - returns success even if state doesn't exist
        """
        # Optionally verify user_id match before deletion
        if user_id:
            # For now, we don't enforce user_id matching
            # In the future, we could check if state.user_id matches before deleting
            pass

        # Delete state (idempotent)
        deleted = await self._state_repo.delete_by_run_id(evaluation_run_id)

        if deleted:
            logger.info(f"Deleted evaluation state for run: {evaluation_run_id}")
        else:
            logger.info(f"No evaluation state found to delete for run: {evaluation_run_id}")
