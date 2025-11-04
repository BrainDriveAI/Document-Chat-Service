import json
from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, ForeignKey, Boolean, Float, select, desc

from ...core.ports.evaluation_repository import EvaluationRepository
from ...core.domain.entities.evaluation import EvaluationRun, EvaluationResult, EvaluationStatus

Base = declarative_base()


# ORM models
class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True)
    collection_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    total_questions = Column(Integer, nullable=False)
    correct_count = Column(Integer, nullable=False, default=0)
    incorrect_count = Column(Integer, nullable=False, default=0)
    run_date = Column(DateTime, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    config_snapshot = Column(JSON, nullable=False)


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"

    id = Column(String, primary_key=True)
    evaluation_run_id = Column(String, ForeignKey("evaluation_runs.id"), nullable=False)
    test_case_id = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    retrieved_context = Column(Text, nullable=False)
    llm_answer = Column(Text, nullable=False)
    judge_correct = Column(Boolean, nullable=False)
    judge_reasoning = Column(Text, nullable=False)
    judge_factual_errors = Column(JSON, nullable=False)
    judge_missing_info = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)


# Repository implementation
class SQLiteEvaluationRepository(EvaluationRepository):
    """SQLite implementation of evaluation repository"""

    def __init__(self, database_url: str):
        """
        Initialize repository with database connection.

        Args:
            database_url: SQLAlchemy database URL (e.g., "sqlite+aiosqlite:///./data/app.db")
        """
        self._engine = create_async_engine(database_url, echo=False, future=True)
        self._async_session = sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            future=True
        )

    async def init_models(self):
        """Create tables if they don't exist"""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_run(self, evaluation_run: EvaluationRun) -> EvaluationRun:
        """Save or update an evaluation run"""
        async with self._async_session() as db_session:
            async with db_session.begin():
                existing = await db_session.get(EvaluationRunModel, evaluation_run.id)
                if existing:
                    # Update existing run
                    existing.status = evaluation_run.status.value
                    existing.total_questions = evaluation_run.total_questions
                    existing.correct_count = evaluation_run.correct_count
                    existing.incorrect_count = evaluation_run.incorrect_count
                    existing.duration_seconds = evaluation_run.duration_seconds
                    existing.config_snapshot = evaluation_run.config_snapshot
                else:
                    # Create new run
                    model = EvaluationRunModel(
                        id=evaluation_run.id,
                        collection_id=evaluation_run.collection_id,
                        status=evaluation_run.status.value,
                        total_questions=evaluation_run.total_questions,
                        correct_count=evaluation_run.correct_count,
                        incorrect_count=evaluation_run.incorrect_count,
                        run_date=evaluation_run.run_date,
                        duration_seconds=evaluation_run.duration_seconds,
                        config_snapshot=evaluation_run.config_snapshot
                    )
                    db_session.add(model)
        return evaluation_run

    async def save_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        """Save an evaluation result"""
        async with self._async_session() as db_session:
            async with db_session.begin():
                model = EvaluationResultModel(
                    id=evaluation_result.id,
                    evaluation_run_id=evaluation_result.evaluation_run_id,
                    test_case_id=evaluation_result.test_case_id,
                    question=evaluation_result.question,
                    ground_truth=evaluation_result.ground_truth,
                    retrieved_context=evaluation_result.retrieved_context,
                    llm_answer=evaluation_result.llm_answer,
                    judge_correct=evaluation_result.judge_correct,
                    judge_reasoning=evaluation_result.judge_reasoning,
                    judge_factual_errors=evaluation_result.judge_factual_errors,
                    judge_missing_info=evaluation_result.judge_missing_info,
                    created_at=evaluation_result.created_at
                )
                db_session.add(model)
        return evaluation_result

    async def find_run_by_id(self, run_id: str) -> Optional[EvaluationRun]:
        """Find evaluation run by ID"""
        async with self._async_session() as db_session:
            result = await db_session.get(EvaluationRunModel, run_id)
            if not result:
                return None

            return EvaluationRun(
                id=result.id,
                collection_id=result.collection_id,
                status=EvaluationStatus(result.status),
                total_questions=result.total_questions,
                correct_count=result.correct_count,
                incorrect_count=result.incorrect_count,
                run_date=result.run_date,
                duration_seconds=result.duration_seconds,
                config_snapshot=result.config_snapshot
            )

    async def find_results_by_run_id(self, run_id: str) -> List[EvaluationResult]:
        """Find all results for an evaluation run"""
        async with self._async_session() as db_session:
            stmt = select(EvaluationResultModel).where(
                EvaluationResultModel.evaluation_run_id == run_id
            ).order_by(EvaluationResultModel.created_at)

            result = await db_session.execute(stmt)
            models = result.scalars().all()

            return [
                EvaluationResult(
                    id=model.id,
                    evaluation_run_id=model.evaluation_run_id,
                    test_case_id=model.test_case_id,
                    question=model.question,
                    ground_truth=model.ground_truth,
                    retrieved_context=model.retrieved_context,
                    llm_answer=model.llm_answer,
                    judge_correct=model.judge_correct,
                    judge_reasoning=model.judge_reasoning,
                    judge_factual_errors=model.judge_factual_errors,
                    judge_missing_info=model.judge_missing_info,
                    created_at=model.created_at
                )
                for model in models
            ]

    async def list_runs(self, limit: int = 50) -> List[EvaluationRun]:
        """List recent evaluation runs"""
        async with self._async_session() as db_session:
            stmt = select(EvaluationRunModel).order_by(
                desc(EvaluationRunModel.run_date)
            ).limit(limit)

            result = await db_session.execute(stmt)
            models = result.scalars().all()

            return [
                EvaluationRun(
                    id=model.id,
                    collection_id=model.collection_id,
                    status=EvaluationStatus(model.status),
                    total_questions=model.total_questions,
                    correct_count=model.correct_count,
                    incorrect_count=model.incorrect_count,
                    run_date=model.run_date,
                    duration_seconds=model.duration_seconds,
                    config_snapshot=model.config_snapshot
                )
                for model in models
            ]
