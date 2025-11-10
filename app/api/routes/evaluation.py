import logging
import re
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, validator

from ...api.deps import (
    get_run_evaluation_use_case,
    get_get_evaluation_results_use_case,
    get_start_plugin_evaluation_use_case,
    get_submit_plugin_evaluation_use_case,
    get_save_evaluation_state_use_case,
    get_load_evaluation_state_use_case,
    get_delete_evaluation_state_use_case,
    get_list_evaluation_states_use_case,
)
from ...core.use_cases.evaluation.run_evaluation import RunEvaluationUseCase
from ...core.use_cases.evaluation.get_results import GetEvaluationResultsUseCase
from ...core.use_cases.evaluation.start_plugin_evaluation import StartPluginEvaluationUseCase
from ...core.use_cases.evaluation.submit_plugin_evaluation import SubmitPluginEvaluationUseCase
from ...core.use_cases.evaluation.save_evaluation_state import SaveEvaluationStateUseCase, StateTooLargeError, InvalidStateSchemaError
from ...core.use_cases.evaluation.load_evaluation_state import LoadEvaluationStateUseCase, StateNotFoundError
from ...core.use_cases.evaluation.delete_evaluation_state import DeleteEvaluationStateUseCase
from ...core.use_cases.evaluation.list_evaluation_states import ListEvaluationStatesUseCase
from ...core.domain.exceptions import EvaluationNotFoundError, EvaluationInitializationError
from ...core.domain.entities.evaluation_config import EvaluationConfig, PersonaConfig, ModelSettings
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


class ModelSettingsRequest(BaseModel):
    """API model for model settings"""
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    context_window: int = 4000
    stop_sequences: List[str] = []


class PersonaConfigRequest(BaseModel):
    """API model for persona configuration"""
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model_settings: ModelSettingsRequest = ModelSettingsRequest()
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_domain(self) -> PersonaConfig:
        """Convert to domain entity"""
        created_at = None
        if self.created_at:
            try:
                created_at = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            except:
                pass

        updated_at = None
        if self.updated_at:
            try:
                updated_at = datetime.fromisoformat(self.updated_at.replace('Z', '+00:00'))
            except:
                pass

        return PersonaConfig(
            id=self.id,
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            model_settings=ModelSettings(
                temperature=self.model_settings.temperature,
                top_p=self.model_settings.top_p,
                frequency_penalty=self.model_settings.frequency_penalty,
                presence_penalty=self.model_settings.presence_penalty,
                context_window=self.model_settings.context_window,
                stop_sequences=self.model_settings.stop_sequences
            ),
            created_at=created_at,
            updated_at=updated_at
        )


class StartWithQuestionsRequest(BaseModel):
    """Request model for starting evaluation with custom questions"""
    collection_id: str
    questions: List[str]
    llm_model: str  # Answer generation model
    persona: Optional[PersonaConfigRequest] = None
    user_id: Optional[str] = None  # Optional user identifier

    @classmethod
    def model_validate(cls, obj):
        """Validate the model"""
        if not obj.get("collection_id"):
            raise ValueError("collection_id is required")
        if not obj.get("questions") or len(obj["questions"]) == 0:
            raise ValueError("questions array is required and must not be empty")
        if not obj.get("llm_model"):
            raise ValueError("llm_model is required")
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


# Background task to submit plugin evaluation
async def submit_evaluation_task(
    submit_plugin_evaluation_use_case: SubmitPluginEvaluationUseCase,
    evaluation_run_id: str,
    submissions: List[Dict[str, Any]]
):
    """Background task to submit and judge evaluation answers"""
    try:
        await submit_plugin_evaluation_use_case.execute_with_questions(
            evaluation_run_id=evaluation_run_id,
            submissions=submissions
        )
        logger.info(f"Background evaluation submission completed for run: {evaluation_run_id}")
    except Exception as e:
        logger.error(f"Background evaluation submission failed: {str(e)}")


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

        # Create domain EvaluationConfig entity
        # Read embedding_model and judge_model from settings
        evaluation_config = EvaluationConfig(
            llm_model=request.llm_model,
            embedding_model=settings.OLLAMA_EMBEDDING_MODEL,
            judge_model=settings.OPENAI_EVALUATION_MODEL,
            persona=request.persona.to_domain() if request.persona else None,
            user_id=request.user_id
        )

        result = await start_plugin_evaluation_use_case.execute_with_questions(
            collection_id=request.collection_id,
            questions=request.questions,
            evaluation_config=evaluation_config
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


@router.post("/plugin/submit-with-questions", status_code=202)
async def submit_plugin_evaluation_with_questions(
    request: SubmitPluginEvaluationRequest,
    background_tasks: BackgroundTasks,
    submit_plugin_evaluation_use_case: SubmitPluginEvaluationUseCase = Depends(get_submit_plugin_evaluation_use_case),
):
    """
    Submit answers for plugin-based evaluation with custom questions.

    Receives LLM answers from plugin, judges them in background, and stores results.
    Loads test cases from database instead of JSON file.
    Supports incremental batch submissions (idempotent).
    Returns immediately (202 Accepted) and processes evaluation in background.
    """
    try:
        logger.info(f"Queuing {len(request.submissions)} answers for evaluation run (with questions): {request.evaluation_run_id}")

        # Validate evaluation run exists
        evaluation_run = await submit_plugin_evaluation_use_case._evaluation_repo.find_run_by_id(request.evaluation_run_id)
        if not evaluation_run:
            raise HTTPException(status_code=404, detail=f"Evaluation run not found: {request.evaluation_run_id}")

        # Convert Pydantic models to dicts
        submissions = [item.model_dump() for item in request.submissions]

        # Add to background tasks
        background_tasks.add_task(
            submit_evaluation_task,
            submit_plugin_evaluation_use_case,
            request.evaluation_run_id,
            submissions
        )

        return {
            "message": "Evaluation submission queued for processing",
            "evaluation_run_id": request.evaluation_run_id,
            "submitted_count": len(submissions)
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to queue evaluation submission: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue evaluation submission: {str(e)}")


# ============================================================================
# Evaluation State Persistence API
# ============================================================================

# Pydantic models for evaluation state endpoints
class ModelInfo(BaseModel):
    """Model information for saved state"""
    id: str
    provider: Literal["ollama", "openai", "anthropic"]
    name: str
    isStreaming: bool


class PersonaInfo(BaseModel):
    """Persona information for saved state"""
    id: str
    name: str
    system_prompt: str


class TestCaseItemState(BaseModel):
    """Test case item in saved state"""
    test_case_id: str
    question: str
    expected_answer: str
    retrieved_context: str
    metadata: Optional[Dict[str, Any]] = None


class BatchItemState(BaseModel):
    """Batch item in saved state"""
    test_case_id: str
    llm_answer: str
    retrieved_context: str


class SaveStateRequest(BaseModel):
    """Request model for saving evaluation state"""
    user_id: Optional[str] = None
    model: ModelInfo
    llm_model: str
    persona: Optional[PersonaInfo] = None
    collection_id: Optional[str] = None
    test_cases: List[TestCaseItemState]
    processed_question_ids: List[str]
    current_batch: List[BatchItemState]
    last_updated: str  # ISO 8601 timestamp

    @validator('user_id')
    def validate_user_id(cls, v):
        if v and not re.match(r'^[0-9a-f]{32}$', v):
            raise ValueError('user_id must be a 32-character hex string')
        return v


class SaveStateResponse(BaseModel):
    """Response model for saving state"""
    success: bool
    state_id: str
    message: str = "Evaluation state saved successfully"


class StateMetadata(BaseModel):
    """Metadata about evaluation state"""
    age_hours: float
    age_days: float
    is_expired: bool
    will_expire_in_hours: float
    backend_evaluation_status: str


class LoadStateResponse(BaseModel):
    """Response model for loading state"""
    state: Dict[str, Any]
    metadata: StateMetadata


class StateSummary(BaseModel):
    """Summary of an evaluation state"""
    run_id: str
    user_id: Optional[str]
    model_name: str
    collection_id: Optional[str]
    total_questions: int
    processed_questions: int
    remaining_questions: int
    last_updated: str
    age_hours: float
    age_days: float
    is_expired: bool
    progress_percentage: float


class ListStatesResponse(BaseModel):
    """Response model for listing states"""
    evaluations: List[StateSummary]
    total_count: int


# Evaluation state endpoints
@router.post("/state/{evaluation_run_id}", response_model=SaveStateResponse)
async def save_evaluation_state(
    evaluation_run_id: str,
    request: SaveStateRequest,
    use_case: SaveEvaluationStateUseCase = Depends(get_save_evaluation_state_use_case)
):
    """
    Save or update evaluation state for resume capability.

    This endpoint allows the plugin to persist evaluation progress for reliable resume.
    """
    try:
        result = await use_case.execute(evaluation_run_id, request.dict())
        return SaveStateResponse(**result)
    except EvaluationNotFoundError as e:
        logger.error(f"Evaluation run not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Evaluation run not found",
                "code": "EVALUATION_RUN_NOT_FOUND",
                "details": {"evaluation_run_id": evaluation_run_id}
            }
        )
    except StateTooLargeError as e:
        logger.error(f"State too large: {str(e)}")
        raise HTTPException(
            status_code=413,
            detail={
                "error": str(e),
                "code": "STATE_TOO_LARGE",
                "details": {"max_size_mb": 10}
            }
        )
    except InvalidStateSchemaError as e:
        logger.error(f"Invalid state schema: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "code": "INVALID_STATE_SCHEMA"
            }
        )
    except Exception as e:
        logger.error(f"Failed to save evaluation state: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save evaluation state: {str(e)}"
        )


@router.get("/state/{evaluation_run_id}", response_model=LoadStateResponse)
async def load_evaluation_state(
    evaluation_run_id: str,
    user_id: Optional[str] = None,
    use_case: LoadEvaluationStateUseCase = Depends(get_load_evaluation_state_use_case)
):
    """
    Load evaluation state for resume.

    Returns state data with metadata including age and expiry status.
    Note: Returns 200 even if state is expired (check metadata.is_expired).
    """
    try:
        result = await use_case.execute(evaluation_run_id, user_id)
        return LoadStateResponse(**result)
    except StateNotFoundError as e:
        logger.error(f"State not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Evaluation state not found",
                "code": "STATE_NOT_FOUND",
                "details": {"evaluation_run_id": evaluation_run_id}
            }
        )
    except Exception as e:
        logger.error(f"Failed to load evaluation state: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load evaluation state: {str(e)}"
        )


@router.delete("/state/{evaluation_run_id}", status_code=204)
async def delete_evaluation_state(
    evaluation_run_id: str,
    user_id: Optional[str] = None,
    use_case: DeleteEvaluationStateUseCase = Depends(get_delete_evaluation_state_use_case)
):
    """
    Delete evaluation state (idempotent).

    Returns 204 No Content even if state doesn't exist.
    """
    try:
        await use_case.execute(evaluation_run_id, user_id)
        return Response(status_code=204)
    except Exception as e:
        logger.error(f"Failed to delete evaluation state: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete evaluation state: {str(e)}"
        )


@router.get("/state/in-progress", response_model=ListStatesResponse)
async def list_in_progress_evaluations(
    user_id: Optional[str] = None,
    include_expired: bool = False,
    use_case: ListEvaluationStatesUseCase = Depends(get_list_evaluation_states_use_case)
):
    """
    List in-progress evaluation states with summaries.

    Query parameters:
    - user_id: Filter by user ID (optional)
    - include_expired: Include states older than 7 days (default: false)
    """
    try:
        result = await use_case.execute(user_id, include_expired)
        return ListStatesResponse(**result)
    except Exception as e:
        logger.error(f"Failed to list evaluation states: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list evaluation states: {str(e)}"
        )
