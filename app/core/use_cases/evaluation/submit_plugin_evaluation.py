import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, UTC

from ...domain.entities.evaluation import EvaluationRun, EvaluationResult, TestCase, EvaluationStatus
from ...ports.evaluation_repository import EvaluationRepository
from ...ports.judge_service import JudgeService

logger = logging.getLogger(__name__)


class SubmitPluginEvaluationUseCase:
    """
    Submit answers for plugin-based evaluation.

    Receives LLM answers from plugin, judges them, and stores results.
    Supports incremental batch submissions (idempotent).
    """

    def __init__(
        self,
        evaluation_repo: EvaluationRepository,
        judge_service: JudgeService,
        test_cases_path: str
    ):
        self._evaluation_repo = evaluation_repo
        self._judge_service = judge_service
        self._test_cases_path = test_cases_path

    async def execute(
        self,
        evaluation_run_id: str,
        submissions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Submit and judge answers for evaluation.

        Args:
            evaluation_run_id: ID of evaluation run
            submissions: List of dicts with:
                - test_case_id: str
                - llm_answer: str
                - retrieved_context: str

        Returns:
            Dict with:
                - evaluation_run_id: str
                - processed_count: int (newly processed)
                - skipped_count: int (already evaluated)
                - total_evaluated: int
                - total_questions: int
                - progress: float (0.0 to 1.0)
                - is_completed: bool
        """
        logger.info(f"Submitting {len(submissions)} answers for evaluation run: {evaluation_run_id}")

        # Get evaluation run
        evaluation_run = await self._evaluation_repo.find_run_by_id(evaluation_run_id)
        if not evaluation_run:
            raise ValueError(f"Evaluation run not found: {evaluation_run_id}")

        # Mark as running if pending
        if evaluation_run.status == EvaluationStatus.PENDING:
            evaluation_run.mark_running()
            await self._evaluation_repo.save_run(evaluation_run)

        # Get already evaluated test case IDs (idempotency)
        evaluated_ids = await self._evaluation_repo.get_evaluated_test_case_ids(evaluation_run_id)
        evaluated_ids_set = set(evaluated_ids)

        logger.info(f"Already evaluated: {len(evaluated_ids_set)} test cases")

        # Load test cases for questions and ground truth
        test_cases_map = self._load_test_cases_map()

        # Process submissions
        processed_count = 0
        skipped_count = 0
        correct_count = 0
        incorrect_count = 0

        for submission in submissions:
            test_case_id = submission["test_case_id"]

            # Skip if already evaluated (idempotent)
            if test_case_id in evaluated_ids_set:
                logger.debug(f"Skipping already evaluated test case: {test_case_id}")
                skipped_count += 1
                continue

            # Get test case details
            test_case = test_cases_map.get(test_case_id)
            if not test_case:
                logger.warning(f"Test case not found: {test_case_id}")
                continue

            # Judge the answer
            logger.debug(f"Judging answer for test case: {test_case_id}")
            judge_output = await self._judge_service.evaluate_answer(
                question=test_case.question,
                retrieved_context=submission["retrieved_context"],
                llm_answer=submission["llm_answer"],
                ground_truth=test_case.ground_truth
            )

            # Create and save result
            evaluation_result = EvaluationResult.create(
                evaluation_run_id=evaluation_run_id,
                test_case_id=test_case_id,
                question=test_case.question,
                ground_truth=test_case.ground_truth,
                retrieved_context=submission["retrieved_context"],
                llm_answer=submission["llm_answer"],
                judge_correct=judge_output.correct,
                judge_reasoning=judge_output.reasoning,
                judge_factual_errors=judge_output.factual_errors,
                judge_missing_info=judge_output.missing_information
            )

            await self._evaluation_repo.save_result(evaluation_result)

            # Update counters
            processed_count += 1
            if judge_output.correct:
                correct_count += 1
            else:
                incorrect_count += 1

            logger.debug(f"Judged test case {test_case_id}: {'CORRECT' if judge_output.correct else 'INCORRECT'}")

        # Update evaluation run
        evaluation_run.evaluated_count = len(evaluated_ids_set) + processed_count
        evaluation_run.correct_count += correct_count
        evaluation_run.incorrect_count += incorrect_count

        # Mark as completed if all questions evaluated
        if evaluation_run.is_completed:
            # Calculate duration (approximate - from run_date to now)
            duration = (datetime.now(UTC) - evaluation_run.run_date).total_seconds()
            evaluation_run.mark_completed(
                correct_count=evaluation_run.correct_count,
                incorrect_count=evaluation_run.incorrect_count,
                duration_seconds=duration
            )
            logger.info(f"Evaluation run {evaluation_run_id} completed")

        await self._evaluation_repo.save_run(evaluation_run)

        logger.info(
            f"Processed {processed_count} new submissions, skipped {skipped_count}. "
            f"Total evaluated: {evaluation_run.evaluated_count}/{evaluation_run.total_questions}"
        )

        return {
            "evaluation_run_id": evaluation_run_id,
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "total_evaluated": evaluation_run.evaluated_count,
            "total_questions": evaluation_run.total_questions,
            "progress": evaluation_run.progress,
            "is_completed": evaluation_run.is_completed,
            "correct_count": evaluation_run.correct_count,
            "incorrect_count": evaluation_run.incorrect_count
        }

    async def execute_with_questions(
        self,
        evaluation_run_id: str,
        submissions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Submit and judge answers for evaluation with custom questions.
        Loads test cases from database instead of JSON file.

        Args:
            evaluation_run_id: ID of evaluation run
            submissions: List of dicts with:
                - test_case_id: str
                - llm_answer: str
                - retrieved_context: str

        Returns:
            Dict with:
                - evaluation_run_id: str
                - processed_count: int (newly processed)
                - skipped_count: int (already evaluated)
                - total_evaluated: int
                - total_questions: int
                - progress: float (0.0 to 1.0)
                - is_completed: bool
        """
        logger.info(f"Submitting {len(submissions)} answers for evaluation run: {evaluation_run_id}")

        # Get evaluation run
        evaluation_run = await self._evaluation_repo.find_run_by_id(evaluation_run_id)
        if not evaluation_run:
            raise ValueError(f"Evaluation run not found: {evaluation_run_id}")

        # Mark as running if pending
        if evaluation_run.status == EvaluationStatus.PENDING:
            evaluation_run.mark_running()
            await self._evaluation_repo.save_run(evaluation_run)

        # Get already evaluated test case IDs (idempotency)
        evaluated_ids = await self._evaluation_repo.get_evaluated_test_case_ids(evaluation_run_id)
        evaluated_ids_set = set(evaluated_ids)

        logger.info(f"Already evaluated: {len(evaluated_ids_set)} test cases")

        # Load test cases from database
        test_cases = await self._evaluation_repo.find_test_cases_by_run_id(evaluation_run_id)
        test_cases_map = {tc.id: tc for tc in test_cases}

        if not test_cases_map:
            raise ValueError(f"No test cases found for evaluation run: {evaluation_run_id}")

        logger.info(f"Loaded {len(test_cases_map)} test cases from database")

        # Process submissions
        processed_count = 0
        skipped_count = 0
        correct_count = 0
        incorrect_count = 0

        for submission in submissions:
            test_case_id = submission["test_case_id"]

            # Skip if already evaluated (idempotent)
            if test_case_id in evaluated_ids_set:
                logger.debug(f"Skipping already evaluated test case: {test_case_id}")
                skipped_count += 1
                continue

            # Get test case details
            test_case = test_cases_map.get(test_case_id)
            if not test_case:
                logger.warning(f"Test case not found: {test_case_id}")
                continue

            # Judge the answer
            logger.debug(f"Judging answer for test case: {test_case_id}")
            judge_output = await self._judge_service.evaluate_answer(
                question=test_case.question,
                retrieved_context=submission["retrieved_context"],
                llm_answer=submission["llm_answer"],
                ground_truth=test_case.ground_truth
            )

            # Create and save result
            evaluation_result = EvaluationResult.create(
                evaluation_run_id=evaluation_run_id,
                test_case_id=test_case_id,
                question=test_case.question,
                ground_truth=test_case.ground_truth,
                retrieved_context=submission["retrieved_context"],
                llm_answer=submission["llm_answer"],
                judge_correct=judge_output.correct,
                judge_reasoning=judge_output.reasoning,
                judge_factual_errors=judge_output.factual_errors,
                judge_missing_info=judge_output.missing_information
            )

            await self._evaluation_repo.save_result(evaluation_result)

            # Update counters
            processed_count += 1
            if judge_output.correct:
                correct_count += 1
            else:
                incorrect_count += 1

            logger.debug(f"Judged test case {test_case_id}: {'CORRECT' if judge_output.correct else 'INCORRECT'}")

        # Update evaluation run
        evaluation_run.evaluated_count = len(evaluated_ids_set) + processed_count
        evaluation_run.correct_count += correct_count
        evaluation_run.incorrect_count += incorrect_count

        # Mark as completed if all questions evaluated
        if evaluation_run.is_completed:
            # Calculate duration (approximate - from run_date to now)
            duration = (datetime.now(UTC) - evaluation_run.run_date).total_seconds()
            evaluation_run.mark_completed(
                correct_count=evaluation_run.correct_count,
                incorrect_count=evaluation_run.incorrect_count,
                duration_seconds=duration
            )
            logger.info(f"Evaluation run {evaluation_run_id} completed")

        await self._evaluation_repo.save_run(evaluation_run)

        logger.info(
            f"Processed {processed_count} new submissions, skipped {skipped_count}. "
            f"Total evaluated: {evaluation_run.evaluated_count}/{evaluation_run.total_questions}"
        )

        return {
            "evaluation_run_id": evaluation_run_id,
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "total_evaluated": evaluation_run.evaluated_count,
            "total_questions": evaluation_run.total_questions,
            "progress": evaluation_run.progress,
            "is_completed": evaluation_run.is_completed,
            "correct_count": evaluation_run.correct_count,
            "incorrect_count": evaluation_run.incorrect_count
        }

    def _load_test_cases_map(self) -> Dict[str, TestCase]:
        """Load test cases and return as dict keyed by ID"""
        test_cases_file = Path(self._test_cases_path)

        if not test_cases_file.exists():
            raise FileNotFoundError(f"Test cases file not found: {self._test_cases_path}")

        with open(test_cases_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = [TestCase.from_dict(tc) for tc in data.get("test_cases", [])]
        return {tc.id: tc for tc in test_cases}
