from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TestCase:
    """Domain entity representing a test case for evaluation"""
    id: str
    question: str
    category: str
    ground_truth: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create TestCase from dictionary (e.g., from test_cases.json)"""
        return cls(
            id=data["id"],
            question=data["question"],
            category=data["category"],
            ground_truth=data.get("ground_truth")
        )


@dataclass
class EvaluationRun:
    """Domain entity representing an evaluation run"""
    id: str
    collection_id: str
    status: EvaluationStatus
    total_questions: int
    correct_count: int
    incorrect_count: int
    run_date: datetime
    duration_seconds: Optional[float]
    config_snapshot: Dict[str, Any]
    evaluated_count: int = 0  # Track how many questions have been evaluated

    @classmethod
    def create(
        cls,
        collection_id: str,
        total_questions: int,
        config_snapshot: Optional[Dict[str, Any]] = None
    ) -> "EvaluationRun":
        """Factory method to create a new evaluation run"""
        return cls(
            id=str(uuid.uuid4()),
            collection_id=collection_id,
            status=EvaluationStatus.PENDING,
            total_questions=total_questions,
            correct_count=0,
            incorrect_count=0,
            evaluated_count=0,
            run_date=datetime.now(UTC),
            duration_seconds=None,
            config_snapshot=config_snapshot or {}
        )

    def mark_running(self):
        """Mark evaluation as running"""
        self.status = EvaluationStatus.RUNNING

    def mark_completed(self, correct_count: int, incorrect_count: int, duration_seconds: float):
        """Mark evaluation as completed"""
        self.status = EvaluationStatus.COMPLETED
        self.correct_count = correct_count
        self.incorrect_count = incorrect_count
        self.duration_seconds = duration_seconds

    def mark_failed(self):
        """Mark evaluation as failed"""
        self.status = EvaluationStatus.FAILED

    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage"""
        if self.total_questions == 0:
            return 0.0
        return (self.correct_count / self.total_questions) * 100

    @property
    def progress(self) -> float:
        """Calculate evaluation progress (0.0 to 1.0)"""
        if self.total_questions == 0:
            return 0.0
        return self.evaluated_count / self.total_questions

    @property
    def is_completed(self) -> bool:
        """Check if all questions have been evaluated"""
        return self.evaluated_count >= self.total_questions


@dataclass
class EvaluationResult:
    """Domain entity representing a single evaluation result"""
    id: str
    evaluation_run_id: str
    test_case_id: str
    question: str
    retrieved_context: str
    llm_answer: str
    judge_correct: bool
    judge_reasoning: str
    judge_factual_errors: List[str]
    judge_missing_info: List[str]
    created_at: datetime
    ground_truth: Optional[str] = None

    @classmethod
    def create(
        cls,
        evaluation_run_id: str,
        test_case_id: str,
        question: str,
        retrieved_context: str,
        llm_answer: str,
        judge_correct: bool,
        judge_reasoning: str,
        judge_factual_errors: List[str],
        judge_missing_info: List[str],
        ground_truth: Optional[str] = None
    ) -> "EvaluationResult":
        """Factory method to create a new evaluation result"""
        return cls(
            id=str(uuid.uuid4()),
            evaluation_run_id=evaluation_run_id,
            test_case_id=test_case_id,
            question=question,
            ground_truth=ground_truth,
            retrieved_context=retrieved_context,
            llm_answer=llm_answer,
            judge_correct=judge_correct,
            judge_reasoning=judge_reasoning,
            judge_factual_errors=judge_factual_errors,
            judge_missing_info=judge_missing_info,
            created_at=datetime.now(UTC)
        )
