from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class ModelSettings:
    """Value object for model settings"""
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    context_window: int = 4000
    stop_sequences: List[str] = None

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "context_window": self.context_window,
            "stop_sequences": self.stop_sequences
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSettings":
        """Create from dictionary"""
        return cls(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            context_window=data.get("context_window", 4000),
            stop_sequences=data.get("stop_sequences", [])
        )


@dataclass
class PersonaConfig:
    """Value object for persona configuration"""
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model_settings: ModelSettings = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.model_settings is None:
            self.model_settings = ModelSettings()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "model_settings": self.model_settings.to_dict(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaConfig":
        """Create from dictionary"""
        created_at = None
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
            else:
                created_at = data["created_at"]

        updated_at = None
        if data.get("updated_at"):
            if isinstance(data["updated_at"], str):
                updated_at = datetime.fromisoformat(data["updated_at"].replace('Z', '+00:00'))
            else:
                updated_at = data["updated_at"]

        return cls(
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            system_prompt=data.get("system_prompt"),
            model_settings=ModelSettings.from_dict(data.get("model_settings", {})),
            created_at=created_at,
            updated_at=updated_at
        )


@dataclass
class EvaluationConfig:
    """Domain entity for evaluation configuration"""
    llm_model: str
    embedding_model: str
    judge_model: str
    persona: Optional[PersonaConfig] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "judge_model": self.judge_model,
            "persona": self.persona.to_dict() if self.persona else None,
            "user_id": self.user_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        """Create from dictionary"""
        persona = None
        if data.get("persona"):
            persona = PersonaConfig.from_dict(data["persona"])

        return cls(
            llm_model=data["llm_model"],
            embedding_model=data["embedding_model"],
            judge_model=data["judge_model"],
            persona=persona,
            user_id=data.get("user_id")
        )
