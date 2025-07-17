from abc import ABC, abstractmethod
from typing import BinaryIO, Optional
from pathlib import Path


class StorageService(ABC):
    """
    Abstract interface for file storage operations.
    Implementations can be local disk, cloud storage, etc.
    """
    
    @abstractmethod
    async def save_file(
        self,
        file_content: BinaryIO,
        collection_id: str,
        filename: str
    ) -> str:
        """
        Save a file to storage.
        
        Args:
            file_content: Binary file content stream
            collection_id: ID of the collection this file belongs to
            filename: Name of the file to save
            
        Returns:
            str: Full path/identifier where the file was saved
        """
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path/identifier of the file to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path/identifier of the file to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Get the size of a file in storage.
        
        Args:
            file_path: Path/identifier of the file
            
        Returns:
            Optional[int]: Size in bytes, None if file doesn't exist
        """
        pass
    
    @abstractmethod
    async def get_file_content(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file content from storage.
        
        Args:
            file_path: Path/identifier of the file
            
        Returns:
            Optional[bytes]: File content, None if file doesn't exist
        """
        pass
