import logging
from typing import Dict, Any, List, Optional

from ...ports.evaluation_state_repository import EvaluationStateRepository

logger = logging.getLogger(__name__)


class ListEvaluationStatesUseCase:
    """Use case for listing evaluation states"""

    def __init__(self, state_repository: EvaluationStateRepository):
        self._state_repo = state_repository

    async def execute(
        self,
        user_id: Optional[str] = None,
        include_expired: bool = False,
        max_age_days: int = 7
    ) -> Dict[str, Any]:
        """
        List in-progress evaluation states with summaries.

        Args:
            user_id: Optional user ID filter
            include_expired: If False, exclude states older than max_age_days
            max_age_days: Maximum age threshold (default 7 days)

        Returns:
            Dictionary with evaluations list and total count
        """
        # Fetch states from repository
        states = await self._state_repo.list_all(
            user_id=user_id,
            include_expired=include_expired,
            max_age_days=max_age_days
        )

        # Build summaries
        summaries = []
        for state in states:
            try:
                # Extract data from state_data JSON
                state_data = state.state_data

                # Extract model name
                model_info = state_data.get("model", {})
                model_name = model_info.get("name", state_data.get("llm_model", "unknown"))

                # Extract collection_id
                collection_id = state_data.get("collection_id")

                # Calculate progress
                test_cases = state_data.get("test_cases", [])
                processed_ids = state_data.get("processed_question_ids", [])
                total_questions = len(test_cases)
                processed_questions = len(processed_ids)
                remaining_questions = total_questions - processed_questions
                progress_percentage = (processed_questions / total_questions * 100) if total_questions > 0 else 0.0

                # Build summary
                summary = {
                    "run_id": state.evaluation_run_id,
                    "user_id": state.user_id,
                    "model_name": model_name,
                    "collection_id": collection_id,
                    "total_questions": total_questions,
                    "processed_questions": processed_questions,
                    "remaining_questions": remaining_questions,
                    "last_updated": state.last_updated.isoformat(),
                    "age_hours": state.age_hours,
                    "age_days": state.age_days,
                    "is_expired": state.is_expired(max_age_days),
                    "progress_percentage": round(progress_percentage, 2)
                }

                summaries.append(summary)

            except Exception as e:
                logger.warning(f"Failed to build summary for state {state.id}: {e}")
                continue

        logger.info(f"Listed {len(summaries)} evaluation states (user_id={user_id}, include_expired={include_expired})")

        return {
            "evaluations": summaries,
            "total_count": len(summaries)
        }
