from dataclasses import dataclass
from typing import List


@dataclass
class JudgeOutput:
    """Domain entity representing the output from the judge LLM evaluation"""
    correct: bool
    reasoning: str
    factual_errors: List[str]
    missing_information: List[str]

    @classmethod
    def create_correct(cls, reasoning: str) -> "JudgeOutput":
        """Factory method for a correct evaluation"""
        return cls(
            correct=True,
            reasoning=reasoning,
            factual_errors=[],
            missing_information=[]
        )

    @classmethod
    def create_incorrect(
        cls,
        reasoning: str,
        factual_errors: List[str],
        missing_information: List[str]
    ) -> "JudgeOutput":
        """Factory method for an incorrect evaluation"""
        return cls(
            correct=False,
            reasoning=reasoning,
            factual_errors=factual_errors,
            missing_information=missing_information
        )
