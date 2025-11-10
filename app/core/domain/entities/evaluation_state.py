from dataclasses import dataclass
from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, Any
import uuid


@dataclass
class EvaluationState:
    """Domain entity representing saved evaluation state for resume capability"""
    id: str
    evaluation_run_id: str
    user_id: Optional[str]
    state_data: Dict[str, Any]
    last_updated: datetime
    created_at: datetime

    @classmethod
    def create(
        cls,
        evaluation_run_id: str,
        state_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> "EvaluationState":
        """Factory method to create a new evaluation state"""
        now = datetime.now(UTC)
        return cls(
            id=str(uuid.uuid4()),
            evaluation_run_id=evaluation_run_id,
            user_id=user_id,
            state_data=state_data,
            last_updated=now,
            created_at=now
        )

    @property
    def age_hours(self) -> float:
        """Calculate age in hours since last update"""
        now = datetime.now(UTC)
        delta = now - self.last_updated
        return delta.total_seconds() / 3600

    @property
    def age_days(self) -> float:
        """Calculate age in days since last update"""
        return self.age_hours / 24

    def is_expired(self, max_age_days: int = 7) -> bool:
        """Check if state has expired (default: 7 days)"""
        return self.age_days > max_age_days

    def will_expire_in_hours(self, max_age_days: int = 7) -> float:
        """Calculate hours until expiration (negative if already expired)"""
        max_age_hours = max_age_days * 24
        return max_age_hours - self.age_hours
