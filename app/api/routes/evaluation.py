import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...api.deps import (
    get_run_evaluation_use_case,
    get_get_evaluation_results_use_case,
    get_start_plugin_evaluation_use_case,
    get_submit_plugin_evaluation_use_case,
)
from ...core.use_cases.evaluation.run_evaluation import RunEvaluationUseCase
from ...core.use_cases.evaluation.get_results import GetEvaluationResultsUseCase
from ...core.use_cases.evaluation.start_plugin_evaluation import StartPluginEvaluationUseCase
from ...core.use_cases.evaluation.submit_plugin_evaluation import SubmitPluginEvaluationUseCase
from ...core.domain.exceptions import EvaluationNotFoundError, EvaluationInitializationError
from ...config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


# Pydantic models for responses
class EvaluationRunResponse(BaseModel):
    """Response model for starting an evaluation"""
    evaluation_id: str
    message: str


class EvaluationStatusResponse(BaseModel):
    """Response model for evaluation status with full results"""
    evaluation_run: Dict[str, Any]
    results: list[Dict[str, Any]]


# Plugin evaluation models
class TestDataItem(BaseModel):
    """Single test data item for plugin evaluation"""
    test_case_id: str
    question: str
    category: str
    retrieved_context: str
    ground_truth: str | None = None


class StartPluginEvaluationResponse(BaseModel):
    """Response model for starting plugin evaluation"""
    evaluation_run_id: str
    test_data: List[TestDataItem]


class SubmissionItem(BaseModel):
    """Single submission item from plugin"""
    test_case_id: str
    llm_answer: str
    retrieved_context: str


class SubmitPluginEvaluationRequest(BaseModel):
    """Request model for submitting plugin evaluation answers"""
    evaluation_run_id: str
    submissions: List[SubmissionItem]


class StartWithQuestionsRequest(BaseModel):
    """Request model for starting evaluation with custom questions"""
    collection_id: str
    questions: List[str]

    @classmethod
    def model_validate(cls, obj):
        """Validate the model"""
        if not obj.get("collection_id"):
            raise ValueError("collection_id is required")
        if not obj.get("questions") or len(obj["questions"]) == 0:
            raise ValueError("questions array is required and must not be empty")
        return super().model_validate(obj)


class SubmitPluginEvaluationResponse(BaseModel):
    """Response model for plugin evaluation submission"""
    evaluation_run_id: str
    processed_count: int
    skipped_count: int
    total_evaluated: int
    total_questions: int
    progress: float
    is_completed: bool
    correct_count: int
    incorrect_count: int


# Background task to run evaluation
async def run_evaluation_task(
    run_evaluation_use_case: RunEvaluationUseCase,
    config_snapshot: Dict[str, Any]
):
    """Background task to run evaluation"""
    try:
        await run_evaluation_use_case.run_evaluation(config_snapshot)
    except Exception as e:
        logger.error(f"Background evaluation task failed: {str(e)}")


@router.post("/run", response_model=EvaluationRunResponse, status_code=202)
async def run_evaluation(
    background_tasks: BackgroundTasks,
    run_evaluation_use_case: RunEvaluationUseCase = Depends(get_run_evaluation_use_case),
):
    """
    Start an evaluation run on the test collection.

    The evaluation runs in the background and processes all test cases.
    Returns immediately with the evaluation_id that can be used to check results.
    """
    try:
        # Create config snapshot
        config_snapshot = {
            "llm_model": settings.OLLAMA_LLM_MODEL,
            "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
            "judge_model": settings.OPENAI_EVALUATION_MODEL,
            "collection_id": settings.EVALUATION_TEST_COLLECTION_ID,
        }

        # Run evaluation synchronously for MVP (can be made async later)
        logger.info("Starting evaluation run")
        evaluation_id = await run_evaluation_use_case.run_evaluation(config_snapshot)

        return EvaluationRunResponse(
            evaluation_id=evaluation_id,
            message="Evaluation completed successfully"
        )

    except EvaluationInitializationError as e:
        logger.error(f"Evaluation initialization failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to run evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run evaluation: {str(e)}")


@router.get("/results/{evaluation_id}", response_model=EvaluationStatusResponse)
async def get_evaluation_results(
    evaluation_id: str,
    get_evaluation_results_use_case: GetEvaluationResultsUseCase = Depends(get_get_evaluation_results_use_case),
):
    """
    Get evaluation results by evaluation ID.

    Returns the evaluation run metadata along with all test case results.
    """
    try:
        results = await get_evaluation_results_use_case.get_evaluation_results(evaluation_id)
        return EvaluationStatusResponse(
            evaluation_run=results["evaluation_run"],
            results=results["results"]
        )
    except EvaluationNotFoundError as e:
        logger.error(f"Evaluation not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get evaluation results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation results: {str(e)}")


@router.get("/runs")
async def list_evaluation_runs(
    limit: int = 50,
    get_evaluation_results_use_case: GetEvaluationResultsUseCase = Depends(get_get_evaluation_results_use_case),
):
    """
    List recent evaluation runs.

    Returns a summary of recent evaluation runs without detailed results.
    """
    try:
        runs = await get_evaluation_results_use_case.list_evaluation_runs(limit=limit)
        return {"runs": runs, "total": len(runs)}
    except Exception as e:
        logger.error(f"Failed to list evaluation runs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list evaluation runs: {str(e)}")


# Plugin evaluation endpoints
@router.post("/plugin/start", response_model=StartPluginEvaluationResponse)
async def start_plugin_evaluation(
    start_plugin_evaluation_use_case: StartPluginEvaluationUseCase = Depends(get_start_plugin_evaluation_use_case),
):
    """
    Start a plugin-based evaluation.

    Creates an evaluation run and retrieves context for all test questions.
    Returns evaluation_run_id and test data (questions + context) for plugin to generate answers.
    """
    try:
        logger.info("Starting plugin evaluation")
        result = await start_plugin_evaluation_use_case.execute()

        return StartPluginEvaluationResponse(
            evaluation_run_id=result["evaluation_run_id"],
            test_data=[TestDataItem(**item) for item in result["test_data"]]
        )

    except FileNotFoundError as e:
        logger.error(f"Test cases file not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start plugin evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start plugin evaluation: {str(e)}")


@router.post("/plugin/start-with-questions", response_model=StartPluginEvaluationResponse)
async def start_plugin_evaluation_with_questions(
    request: StartWithQuestionsRequest,
    start_plugin_evaluation_use_case: StartPluginEvaluationUseCase = Depends(get_start_plugin_evaluation_use_case),
):
    """
    Start a plugin-based evaluation with custom questions.

    Creates an evaluation run and retrieves context for provided questions.
    Returns evaluation_run_id and test data (questions + context) for plugin to generate answers.
    """
    try:
        logger.info(f"Starting plugin evaluation with {len(request.questions)} custom questions for collection {request.collection_id}")

        result = await start_plugin_evaluation_use_case.execute_with_questions(
            collection_id=request.collection_id,
            questions=request.questions
        )

        return StartPluginEvaluationResponse(
            evaluation_run_id=result["evaluation_run_id"],
            test_data=[TestDataItem(**item) for item in result["test_data"]]
        )

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        # Check if it's a collection not found error
        if "Collection not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start plugin evaluation with questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start plugin evaluation: {str(e)}")


@router.post("/plugin/submit", response_model=SubmitPluginEvaluationResponse)
async def submit_plugin_evaluation(
    request: SubmitPluginEvaluationRequest,
    submit_plugin_evaluation_use_case: SubmitPluginEvaluationUseCase = Depends(get_submit_plugin_evaluation_use_case),
):
    """
    Submit answers for plugin-based evaluation.

    Receives LLM answers from plugin, judges them, and stores results.
    Supports incremental batch submissions (idempotent).
    """
    try:
        logger.info(f"Submitting {len(request.submissions)} answers for evaluation run: {request.evaluation_run_id}")

        # Convert Pydantic models to dicts
        submissions = [item.model_dump() for item in request.submissions]

        result = await submit_plugin_evaluation_use_case.execute(
            evaluation_run_id=request.evaluation_run_id,
            submissions=submissions
        )

        return SubmitPluginEvaluationResponse(**result)

    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Test cases file not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit plugin evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit plugin evaluation: {str(e)}")
