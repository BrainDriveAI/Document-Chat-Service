import logging
from typing import List, Dict, Any

from ...domain.entities.evaluation import EvaluationRun, EvaluationResult
from ...domain.exceptions import EvaluationNotFoundError
from ...ports.evaluation_repository import EvaluationRepository

logger = logging.getLogger(__name__)


class GetEvaluationResultsUseCase:
    """Use case for retrieving evaluation results"""

    def __init__(self, evaluation_repo: EvaluationRepository):
        self.evaluation_repo = evaluation_repo

    async def get_evaluation_results(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get evaluation run with all results.

        Args:
            evaluation_id: ID of the evaluation run

        Returns:
            Dictionary containing evaluation run and results

        Raises:
            EvaluationNotFoundError: If evaluation run not found
        """
        # Get evaluation run
        evaluation_run = await self.evaluation_repo.find_run_by_id(evaluation_id)
        if not evaluation_run:
            raise EvaluationNotFoundError(f"Evaluation run {evaluation_id} not found")

        # Get all results for this run
        results = await self.evaluation_repo.find_results_by_run_id(evaluation_id)

        logger.info(f"Retrieved evaluation {evaluation_id} with {len(results)} results")

        return {
            "evaluation_run": {
                "id": evaluation_run.id,
                "collection_id": evaluation_run.collection_id,
                "status": evaluation_run.status.value,
                "total_questions": evaluation_run.total_questions,
                "correct_count": evaluation_run.correct_count,
                "incorrect_count": evaluation_run.incorrect_count,
                "accuracy": evaluation_run.accuracy,
                "run_date": evaluation_run.run_date.isoformat(),
                "duration_seconds": evaluation_run.duration_seconds,
                "config_snapshot": evaluation_run.config_snapshot
            },
            "results": [
                {
                    "id": result.id,
                    "test_case_id": result.test_case_id,
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    "retrieved_context": result.retrieved_context,
                    "llm_answer": result.llm_answer,
                    "judge_correct": result.judge_correct,
                    "judge_reasoning": result.judge_reasoning,
                    "judge_factual_errors": result.judge_factual_errors,
                    "judge_missing_info": result.judge_missing_info,
                    "created_at": result.created_at.isoformat()
                }
                for result in results
            ]
        }

    async def list_evaluation_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent evaluation runs.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of evaluation run summaries
        """
        runs = await self.evaluation_repo.list_runs(limit=limit)

        logger.info(f"Retrieved {len(runs)} evaluation runs")

        return [
            {
                "id": run.id,
                "collection_id": run.collection_id,
                "status": run.status.value,
                "total_questions": run.total_questions,
                "correct_count": run.correct_count,
                "incorrect_count": run.incorrect_count,
                "accuracy": run.accuracy,
                "run_date": run.run_date.isoformat(),
                "duration_seconds": run.duration_seconds
            }
            for run in runs
        ]
