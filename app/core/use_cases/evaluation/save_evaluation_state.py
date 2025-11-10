import json
import re
import logging
from typing import Dict, Any
from datetime import datetime, UTC

from ...ports.evaluation_state_repository import EvaluationStateRepository
from ...ports.evaluation_repository import EvaluationRepository
from ...domain.entities.evaluation_state import EvaluationState
from ...domain.exceptions import EvaluationNotFoundError

logger = logging.getLogger(__name__)

# Max state size: 10 MB
MAX_STATE_SIZE_BYTES = 10 * 1024 * 1024


class StateTooLargeError(Exception):
    """Raised when state data exceeds maximum size"""
    pass


class InvalidStateSchemaError(Exception):
    """Raised when state data fails validation"""
    pass


class SaveEvaluationStateUseCase:
    """Use case for saving evaluation state"""

    def __init__(
        self,
        state_repository: EvaluationStateRepository,
        evaluation_repository: EvaluationRepository
    ):
        self._state_repo = state_repository
        self._eval_repo = evaluation_repository

    async def execute(self, evaluation_run_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save evaluation state for resume capability.

        Args:
            evaluation_run_id: The evaluation run ID
            request_data: State data from request

        Returns:
            Dictionary with success flag and state_id

        Raises:
            EvaluationNotFoundError: If evaluation run doesn't exist
            StateTooLargeError: If state data exceeds 10 MB
            InvalidStateSchemaError: If state data fails validation
        """
        # Validate evaluation run exists
        evaluation_run = await self._eval_repo.find_run_by_id(evaluation_run_id)
        if not evaluation_run:
            raise EvaluationNotFoundError(f"Evaluation run not found: {evaluation_run_id}")

        # Extract user_id if provided
        user_id = request_data.get("user_id")

        # Validate user_id format if provided
        if user_id and not re.match(r'^[0-9a-f]{32}$', user_id):
            raise InvalidStateSchemaError("user_id must be a 32-character hex string")

        # Validate state size
        state_json = json.dumps(request_data)
        state_size = len(state_json.encode('utf-8'))

        if state_size > MAX_STATE_SIZE_BYTES:
            size_mb = state_size / (1024 * 1024)
            raise StateTooLargeError(
                f"State size ({size_mb:.2f} MB) exceeds maximum allowed (10 MB)"
            )

        # Validate required fields
        required_fields = ['model', 'llm_model', 'test_cases', 'processed_question_ids']
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            raise InvalidStateSchemaError(f"Missing required fields: {', '.join(missing_fields)}")

        # Parse last_updated timestamp if provided
        last_updated_str = request_data.get("last_updated")
        if last_updated_str:
            try:
                last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # If parsing fails, use current time
                last_updated = datetime.now(UTC)
        else:
            last_updated = datetime.now(UTC)

        # Create or update state entity
        state = EvaluationState.create(
            evaluation_run_id=evaluation_run_id,
            state_data=request_data,
            user_id=user_id
        )
        # Override last_updated if provided in request
        state.last_updated = last_updated

        # Save state (upsert)
        saved_state = await self._state_repo.save_state(state)

        logger.info(f"Saved evaluation state for run: {evaluation_run_id}, size: {state_size} bytes")

        return {
            "success": True,
            "state_id": saved_state.id,
            "message": "Evaluation state saved successfully"
        }
