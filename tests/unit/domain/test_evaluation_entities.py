"""
Unit tests for evaluation domain entities (EvaluationRun, EvaluationResult, TestCase).

Tests cover:
- Factory methods
- State transitions
- Progress calculations
- Accuracy calculations
"""

import pytest
from datetime import datetime
from uuid import UUID

from app.core.domain.entities.evaluation import (
    EvaluationRun,
    EvaluationStatus,
    TestCase
)


class TestTestCase:
    """Tests for TestCase entity."""

    def test_create_test_case_from_dict(self):
        """Test creating TestCase from dictionary."""
        data = {
            "id": "tc-1",
            "question": "What is AI?",
            "category": "general",
            "ground_truth": "Artificial Intelligence is..."
        }

        test_case = TestCase.from_dict(data)

        assert test_case.id == "tc-1"
        assert test_case.question == "What is AI?"
        assert test_case.category == "general"
        assert test_case.ground_truth == "Artificial Intelligence is..."

    def test_create_test_case_without_ground_truth(self):
        """Test creating TestCase without ground truth."""
        data = {
            "id": "tc-2",
            "question": "What is ML?",
            "category": "technical"
        }

        test_case = TestCase.from_dict(data)

        assert test_case.ground_truth is None


class TestEvaluationRun:
    """Tests for EvaluationRun entity."""

    def test_create_evaluation_run(self):
        """Test creating a new evaluation run."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10,
            config_snapshot={"model": "llama3", "temperature": 0.7},
            user_id="user-456"
        )

        assert run.id is not None
        assert UUID(run.id)  # Valid UUID
        assert run.collection_id == "collection-123"
        assert run.status == EvaluationStatus.PENDING
        assert run.total_questions == 10
        assert run.correct_count == 0
        assert run.incorrect_count == 0
        assert run.evaluated_count == 0
        assert run.duration_seconds is None
        assert run.config_snapshot == {"model": "llama3", "temperature": 0.7}
        assert run.user_id == "user-456"
        assert isinstance(run.run_date, datetime)

    def test_create_evaluation_run_defaults(self):
        """Test creating run with default config and no user_id."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=5
        )

        assert run.config_snapshot == {}
        assert run.user_id is None

    def test_mark_running(self):
        """Test marking evaluation as running."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )

        run.mark_running()

        assert run.status == EvaluationStatus.RUNNING

    def test_mark_completed(self):
        """Test marking evaluation as completed."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.mark_running()

        run.mark_completed(
            correct_count=8,
            incorrect_count=2,
            duration_seconds=120.5
        )

        assert run.status == EvaluationStatus.COMPLETED
        assert run.correct_count == 8
        assert run.incorrect_count == 2
        assert run.duration_seconds == 120.5

    def test_mark_failed(self):
        """Test marking evaluation as failed."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.mark_running()

        run.mark_failed()

        assert run.status == EvaluationStatus.FAILED

    def test_accuracy_calculation(self):
        """Test accuracy property calculation."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.mark_completed(correct_count=7, incorrect_count=3, duration_seconds=100.0)

        assert run.accuracy == 70.0  # 7/10 = 70%

    def test_accuracy_zero_questions(self):
        """Test accuracy when total_questions is 0."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=0
        )

        assert run.accuracy == 0.0

    def test_accuracy_all_correct(self):
        """Test accuracy when all answers are correct."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.mark_completed(correct_count=10, incorrect_count=0, duration_seconds=100.0)

        assert run.accuracy == 100.0

    def test_accuracy_all_incorrect(self):
        """Test accuracy when all answers are incorrect."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.mark_completed(correct_count=0, incorrect_count=10, duration_seconds=100.0)

        assert run.accuracy == 0.0

    def test_progress_calculation(self):
        """Test progress property calculation."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.evaluated_count = 5

        assert run.progress == 0.5  # 5/10 = 0.5

    def test_progress_zero_questions(self):
        """Test progress when total_questions is 0."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=0
        )

        assert run.progress == 0.0

    def test_progress_complete(self):
        """Test progress when all questions evaluated."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=10
        )
        run.evaluated_count = 10

        assert run.progress == 1.0

    def test_progress_partial(self):
        """Test progress at various completion stages."""
        run = EvaluationRun.create(
            collection_id="collection-123",
            total_questions=20
        )

        # 25% complete
        run.evaluated_count = 5
        assert run.progress == 0.25

        # 75% complete
        run.evaluated_count = 15
        assert run.progress == 0.75

        # 90% complete
        run.evaluated_count = 18
        assert run.progress == 0.9
