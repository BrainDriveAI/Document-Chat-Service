from abc import ABC, abstractmethod
from typing import Any

class Logger(ABC):
    """Abstract logger interface"""
    
    @abstractmethod
    def debug(self, message: str, *args: Any) -> None:
        pass
    
    @abstractmethod
    def info(self, message: str, *args: Any) -> None:
        pass
    
    @abstractmethod
    def warning(self, message: str, *args: Any) -> None:
        pass
    
    @abstractmethod
    def error(self, message: str, *args: Any) -> None:
        pass
    
    @abstractmethod
    def critical(self, message: str, *args: Any) -> None:
        pass
