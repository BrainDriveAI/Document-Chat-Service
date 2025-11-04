import logging
import time
from typing import Dict, Any

from ...domain.entities.evaluation import EvaluationRun, EvaluationResult
from ...domain.exceptions import EvaluationInitializationError
from ...ports.evaluation_repository import EvaluationRepository
from ...ports.judge_service import JudgeService
from ...ports.llm_service import LLMService
from ..context_retrieval import ContextRetrievalUseCase
from .initialize_test_collection import InitializeTestCollectionUseCase

logger = logging.getLogger(__name__)


class RunEvaluationUseCase:
    """Use case for running evaluation on test collection"""

    def __init__(
        self,
        evaluation_repo: EvaluationRepository,
        judge_service: JudgeService,
        llm_service: LLMService,
        context_retrieval_use_case: ContextRetrievalUseCase,
        initialize_test_collection_use_case: InitializeTestCollectionUseCase,
        test_collection_id: str
    ):
        self.evaluation_repo = evaluation_repo
        self.judge_service = judge_service
        self.llm_service = llm_service
        self.context_retrieval_use_case = context_retrieval_use_case
        self.initialize_test_collection_use_case = initialize_test_collection_use_case
        self.test_collection_id = test_collection_id

    async def run_evaluation(self, config_snapshot: Dict[str, Any]) -> str:
        """
        Run evaluation on all test cases.

        Args:
            config_snapshot: Configuration snapshot for this evaluation run

        Returns:
            Evaluation run ID

        Raises:
            EvaluationInitializationError: If test cases cannot be loaded
        """
        start_time = time.time()

        try:
            # Load test cases
            logger.info("Loading test cases for evaluation")
            test_cases = await self.initialize_test_collection_use_case.load_test_cases()
            logger.info(f"Loaded {len(test_cases)} test cases")

            if not test_cases:
                raise EvaluationInitializationError("No test cases found")

            # Create evaluation run
            evaluation_run = EvaluationRun.create(
                collection_id=self.test_collection_id,
                total_questions=len(test_cases),
                config_snapshot=config_snapshot
            )

            # Save initial run
            evaluation_run.mark_running()
            await self.evaluation_repo.save_run(evaluation_run)
            logger.info(f"Created evaluation run: {evaluation_run.id}")

            # Process each test case
            correct_count = 0
            incorrect_count = 0

            for idx, test_case in enumerate(test_cases, 1):
                logger.info(f"Processing test case {idx}/{len(test_cases)}: {test_case.id}")

                try:
                    # Step 1: Retrieve context
                    context_result = await self.context_retrieval_use_case.retrieve_context(
                        query_text=test_case.question,
                        collection_id=self.test_collection_id,
                        top_k=5,
                        use_hybrid=True,
                        use_intent_classification=False,  # Skip intent classification for evaluation
                        query_transformation_enabled=False  # Skip query transformation for evaluation
                    )

                    # Combine retrieved chunks into context string
                    retrieved_context = "\n\n".join([
                        f"[Chunk {i+1}]\n{chunk.content}"
                        for i, chunk in enumerate(context_result.chunks)
                    ])

                    if not retrieved_context:
                        retrieved_context = "[No relevant context found]"

                    # Step 2: Generate answer using LLM
                    llm_answer = await self.llm_service.generate_response(
                        prompt=test_case.question,
                        context=retrieved_context,
                        temperature=0.1
                    )

                    # Step 3: Evaluate answer using judge
                    judge_output = await self.judge_service.evaluate_answer(
                        question=test_case.question,
                        retrieved_context=retrieved_context,
                        llm_answer=llm_answer,
                        ground_truth=test_case.ground_truth
                    )

                    # Step 4: Create and save result
                    evaluation_result = EvaluationResult.create(
                        evaluation_run_id=evaluation_run.id,
                        test_case_id=test_case.id,
                        question=test_case.question,
                        ground_truth=test_case.ground_truth,
                        retrieved_context=retrieved_context,
                        llm_answer=llm_answer,
                        judge_correct=judge_output.correct,
                        judge_reasoning=judge_output.reasoning,
                        judge_factual_errors=judge_output.factual_errors,
                        judge_missing_info=judge_output.missing_information
                    )

                    await self.evaluation_repo.save_result(evaluation_result)

                    # Update counts
                    if judge_output.correct:
                        correct_count += 1
                        logger.info(f"Test case {test_case.id}: CORRECT")
                    else:
                        incorrect_count += 1
                        logger.info(f"Test case {test_case.id}: INCORRECT")

                except Exception as e:
                    logger.error(f"Failed to process test case {test_case.id}: {str(e)}")
                    # Mark as incorrect if processing fails
                    incorrect_count += 1
                    continue

            # Calculate duration
            duration_seconds = time.time() - start_time

            # Mark run as completed
            evaluation_run.mark_completed(
                correct_count=correct_count,
                incorrect_count=incorrect_count,
                duration_seconds=duration_seconds
            )
            await self.evaluation_repo.save_run(evaluation_run)

            logger.info(
                f"Evaluation completed: {correct_count}/{len(test_cases)} correct "
                f"({evaluation_run.accuracy:.1f}%) in {duration_seconds:.1f}s"
            )

            return evaluation_run.id

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            # Try to mark run as failed if it was created
            try:
                if 'evaluation_run' in locals():
                    evaluation_run.mark_failed()
                    await self.evaluation_repo.save_run(evaluation_run)
            except:
                pass
            raise
