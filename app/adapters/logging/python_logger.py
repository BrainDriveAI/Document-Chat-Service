import logging
from ...core.ports.logger import Logger

class PythonLogger(Logger):
    """Implementation using Python's logging module"""
    
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
    
    def debug(self, message: str, *args) -> None:
        self._logger.debug(message, *args)
    
    def info(self, message: str, *args) -> None:
        self._logger.info(message, *args)
    
    def warning(self, message: str, *args) -> None:
        self._logger.warning(message, *args)
    
    def error(self, message: str, *args) -> None:
        self._logger.error(message, *args)
    
    def critical(self, message: str, *args) -> None:
        self._logger.critical(message, *args)
