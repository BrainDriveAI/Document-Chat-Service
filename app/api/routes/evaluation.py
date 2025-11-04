import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...api.deps import (
    get_run_evaluation_use_case,
    get_get_evaluation_results_use_case,
)
from ...core.use_cases.evaluation.run_evaluation import RunEvaluationUseCase
from ...core.use_cases.evaluation.get_results import GetEvaluationResultsUseCase
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
