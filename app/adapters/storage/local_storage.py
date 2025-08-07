import uuid
import shutil
import aiofiles
from pathlib import Path
from typing import BinaryIO, Optional
from ...core.ports.storage_service import StorageService
from ...config import settings


class LocalStorageService(StorageService):
    """
    Local disk storage implementation.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize local storage service.
        
        Args:
            base_path: Base directory for storing files. If None, uses settings.UPLOADS_DIR
        """
        self.base_path = Path(base_path or settings.UPLOADS_DIR)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_file(
        self,
        file_content: BinaryIO,
        collection_id: str,
        filename: str
    ) -> str:
        """
        Save file to local disk storage.
        
        Args:
            file_content: Binary file content stream
            collection_id: ID of the collection this file belongs to
            filename: Original filename
            
        Returns:
            str: Full path where the file was saved
        """
        # Create collection directory
        collection_dir = self.base_path / collection_id
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename while preserving extension
        file_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        saved_filename = f"{file_id}{ext}"
        saved_path = collection_dir / saved_filename
        
        try:
            # Save file asynchronously
            async with aiofiles.open(saved_path, "wb") as out_file:
                file_content.seek(0)  # Reset to beginning
                content = file_content.read()
                await out_file.write(content)
            
            return str(saved_path)
            
        except Exception as e:
            # Cleanup on failure
            if saved_path.exists():
                saved_path.unlink(missing_ok=True)
            raise e
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from local storage.
        
        Args:
            file_path: Path of the file to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception:
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in local storage.
        
        Args:
            file_path: Path of the file to check
            
        Returns:
            bool: True if file exists
        """
        try:
            return Path(file_path).exists()
        except Exception:
            return False
    
    async def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Get file size from local storage.
        
        Args:
            file_path: Path of the file
            
        Returns:
            Optional[int]: Size in bytes, None if file doesn't exist
        """
        try:
            path = Path(file_path)
            if path.exists():
                return path.stat().st_size
            return None
        except Exception:
            return None
    
    async def get_file_content(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file content from local storage.
        
        Args:
            file_path: Path of the file
            
        Returns:
            Optional[bytes]: File content, None if file doesn't exist
        """
        try:
            path = Path(file_path)
            if path.exists():
                async with aiofiles.open(path, "rb") as f:
                    return await f.read()
            return None
        except Exception:
            return None
