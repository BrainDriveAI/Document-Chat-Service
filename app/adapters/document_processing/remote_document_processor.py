import aiohttp
import asyncio
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from ...core.ports.document_processor import DocumentProcessor
from ...core.ports.storage_service import StorageService
from ...core.domain.entities.document import Document
from ...core.domain.entities.document_chunk import DocumentChunk
from ...core.domain.exceptions import DocumentProcessingError


class RemoteDocumentProcessor(DocumentProcessor):
    """
    Remote document processor that calls deployed document processing API.
    """
    
    def __init__(
            self, 
            api_base_url: str,
            storage_service: StorageService,
            timeout: int = 300,
            max_retries: int = 1,
            api_key: Optional[str] = None,
        ):
        """
        Initialize remote document processor.
        
        Args:
            api_base_url: Base URL of the deployed document processing API
            storage_service: Storage service to read document files
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.storage_service = storage_service
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        
        # Supported document types (should match your deployed API)
        self.supported_types = ['pdf', 'docx', 'doc']
    
    # async def process_document(self, document: Document) -> Tuple[List[DocumentChunk], str]:
    #     """
    #     Process a document using the remote API.
        
    #     Args:
    #         document: Document entity to process
            
    #     Returns:
    #         Tuple of (document chunks, complete text)
    #     """
    #     try:
    #         self.logger.info(f"Starting remote processing for document {document.id}")
            
    #         # Get file content from storage
    #         file_content = await self.storage_service.get_file_content(document.file_path)
    #         if file_content is None:
    #             raise DocumentProcessingError(f"Could not retrieve file content for document {document.id}")
            
    #         # Prepare file for upload
    #         # file_data = {
    #         #     'file': (document.original_filename, file_content, self._get_content_type(document.original_filename))
    #         # }
            
    #         # Make API call with retries
    #         response_data = await self._make_api_call_with_retry(file_content, document)
            
    #         # Parse response and convert to domain objects
    #         doc_chunks, complete_text = self._parse_api_response(response_data, document)
            
    #         self.logger.info(f"Successfully processed document {document.id} remotely, got {len(doc_chunks)} chunks")
    #         return doc_chunks, complete_text
            
    #     except Exception as e:
    #         self.logger.error(f"Remote document processing failed for {document.id}: {str(e)}")
    #         raise DocumentProcessingError(f"Remote processing failed: {str(e)}")
    
    def get_supported_types(self) -> List[str]:
        """Return list of supported document types"""
        return self.supported_types.copy()
    
    async def _make_api_call_with_retry(self, file_content: bytes, document: Document) -> dict:
        """
        Make API call with retry logic.
        
        Args:
            file_content: File content as bytes
            document: Document entity for metadata
            
        Returns:
            API response data
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting API call for document {document.id} (attempt {attempt + 1}/{self.max_retries})")
                
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Create form data using aiohttp.FormData
                    form_data = aiohttp.FormData()
                    form_data.add_field(
                        'file',
                        file_content,
                        filename=document.original_filename,
                        content_type=self._get_content_type(document.original_filename)
                    )
                    
                    async with session.post(
                        f"{self.api_base_url}/upload",
                        data=form_data
                    ) as response:
                        self.logger.debug(f"Response status: {response}")
                        if response.status == 200:
                            response_data = await response.json()
                            self.logger.info(f"Successfully called remote API for document {document.id}")
                            self.logger.debug(f"Response data: {response_data}")
                            return response_data
                        else:
                            error_text = await response.text()
                            raise DocumentProcessingError(
                                f"API returned status {response.status}: {error_text}"
                            )
                            
            except asyncio.TimeoutError as e:
                last_exception = DocumentProcessingError(f"API call timed out after {self.timeout} seconds")
                self.logger.warning(f"API call timeout for document {document.id} (attempt {attempt + 1})")
                
            except aiohttp.ClientError as e:
                last_exception = DocumentProcessingError(f"API client error: {str(e)}")
                self.logger.warning(f"API client error for document {document.id} (attempt {attempt + 1}): {str(e)}")
                
            except Exception as e:
                last_exception = DocumentProcessingError(f"Unexpected API error: {str(e)}")
                self.logger.error(f"Unexpected API error for document {document.id} (attempt {attempt + 1}): {str(e)}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
        
        # All retries failed
        raise last_exception or DocumentProcessingError("All retry attempts failed")
    
    def _parse_api_response(self, response_data: dict, document: Document) -> Tuple[List[DocumentChunk], str]:
        """
        Parse API response and convert to domain objects.
        
        Args:
            response_data: Response data from API (tuple format: [chunks_list, complete_text])
            document: Original document entity
            
        Returns:
            Tuple of (document chunks, complete text)
        """
        try:
            # API 1 returns a tuple: [chunks_list, complete_text]
            # Handle both tuple and dict formats for flexibility
            if isinstance(response_data, list) and len(response_data) == 2:
                chunks_data = response_data[0]
                complete_text = response_data[1]
            elif isinstance(response_data, dict):
                chunks_data = response_data.get('chunks', [])
                complete_text = response_data.get('complete_text', '')
            else:
                raise DocumentProcessingError(f"Unexpected response format: {type(response_data)}")
            
            # Convert API response chunks to DocumentChunk entities
            doc_chunks = []
            for i, chunk_data in enumerate(chunks_data):
                # API 1 chunk structure uses different field names
                chunk = DocumentChunk(
                    id=chunk_data.get('id', f"{document.id}_chunk_{i}"),
                    document_id=document.id,
                    collection_id=getattr(document, 'collection_id', 'default_collection'),
                    chunk_index=chunk_data.get('chunk_index', i),
                    content=chunk_data.get('content', ''),
                    chunk_type='paragraph',  # Default since API 1 doesn't specify this
                    parent_chunk_id=None,   # API 1 doesn't have hierarchical chunking
                    metadata={
                        # Map API 1 metadata structure to API 2 format
                        'created_at': chunk_data.get('created_at'),
                        'start_char': chunk_data.get('start_char', 0),
                        'end_char': chunk_data.get('end_char', 0),
                        'token_count': chunk_data.get('token_count', 0),
                        
                        # Extract nested metadata from API 1
                        **self._extract_api1_metadata(chunk_data.get('metadata', {}))
                    }
                )
                doc_chunks.append(chunk)
            
            return doc_chunks, complete_text
            
        except Exception as e:
            self.logger.error(f"Failed to parse API response: {str(e)}")
            raise DocumentProcessingError(f"Failed to parse API response: {str(e)}")
    
    def _get_content_type(self, filename: str) -> str:
        """
        Get content type based on file extension.
        
        Args:
            filename: Name of the file
            
        Returns:
            Content type string
        """
        # ext = Path(filename).suffix.lower()
        # content_types = {
        #     '.pdf': 'application/pdf',
        #     '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        #     '.doc': 'application/msword'
        # }
        # return content_types.get(ext, 'application/octet-stream')
        return "application/octet-stream"

    def _parse_processing_service_response(self, response_data: dict, document: Document) -> Tuple[List[DocumentChunk], str]:
        """
        Parse document processing service response and convert to domain objects.
        
        Args:
            response_data: Response data from document processing service (tuple format: [chunks_list, complete_text])
            document: Original document entity
            
        Returns:
            Tuple of (document chunks, complete text)
        """
        try:
            # Document processing service returns a tuple: [chunks_list, complete_text]
            # Handle both tuple and dict formats for flexibility
            if isinstance(response_data, list) and len(response_data) == 2:
                processing_service_chunks = response_data[0]
                extracted_text = response_data[1]
            elif isinstance(response_data, dict):
                processing_service_chunks = response_data.get('chunks', [])
                extracted_text = response_data.get('complete_text', '')
            else:
                raise DocumentProcessingError(f"Unexpected response format: {type(response_data)}")
            
            # Convert processing service chunks to local DocumentChunk entities
            local_chunks = []
            for chunk_index, processing_chunk in enumerate(processing_service_chunks):
                # Transform processing service chunk structure to local format
                local_chunk = DocumentChunk(
                    id=processing_chunk.get('id', f"{document.id}_chunk_{chunk_index}"),
                    document_id=document.id,
                    collection_id=getattr(document, 'collection_id', 'default_collection'),
                    chunk_index=processing_chunk.get('chunk_index', chunk_index),
                    content=processing_chunk.get('content', ''),
                    chunk_type='paragraph',  # Default since processing service doesn't specify this
                    parent_chunk_id=None,   # Processing service doesn't have hierarchical chunking
                    metadata={
                        # Map processing service metadata to local format
                        'created_at': processing_chunk.get('created_at'),
                        'start_char': processing_chunk.get('start_char', 0),
                        'end_char': processing_chunk.get('end_char', 0),
                        'token_count': processing_chunk.get('token_count', 0),
                        
                        # Extract nested metadata from processing service
                        **self._transform_processing_service_metadata(processing_chunk.get('metadata', {}))
                    }
                )
                local_chunks.append(local_chunk)
            
            return local_chunks, extracted_text
            
        except Exception as e:
            self.logger.error(f"Failed to parse processing service response: {str(e)}")
            raise DocumentProcessingError(f"Failed to parse processing service response: {str(e)}")

    def _transform_processing_service_metadata(self, processing_service_metadata: dict) -> dict:
        """
        Transform metadata from processing service format to local format.
        
        Args:
            processing_service_metadata: Metadata dict from document processing service
            
        Returns:
            Transformed metadata dict for local use
        """
        # Processing service metadata fields (based on spaCy layout processor ChunkMetadata)
        local_metadata = {}
        
        # Direct field mappings from processing service to local format
        field_mappings = {
            'document_filename': 'document_filename',
            'document_type': 'document_type',
            'chunk_token_count': 'chunk_token_count',
            'chunk_char_count': 'chunk_char_count',
            'processing_method': 'processing_method',
            'language': 'language',
            'section_title': 'section_title',
            'section_level': 'section_level',
            'page_number': 'page_number',
            'paragraph_index': 'paragraph_index',
            'bbox': 'bbox',
            'font_size': 'font_size',
            'font_family': 'font_family',
            'text_color': 'text_color',
            'content_type': 'content_type',
            'topics': 'topics',
            'entities': 'entities',
            'chunking_strategy': 'chunking_strategy',
            'chunk_method': 'chunk_method',
            'overlap_with_previous': 'overlap_with_previous',
            'overlap_with_next': 'overlap_with_next'
        }
        
        for processing_field, local_field in field_mappings.items():
            if processing_field in processing_service_metadata:
                local_metadata[local_field] = processing_service_metadata[processing_field]
        
        # Add custom fields if they exist in processing service response
        if 'custom_fields' in processing_service_metadata:
            local_metadata.update(processing_service_metadata['custom_fields'])
        
        return local_metadata

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers if API key is configured"""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _call_processing_service_with_retry(self, file_content: bytes, document: Document) -> dict:
        """
        Call document processing service with retry logic.
        
        Args:
            file_content: File content as bytes
            document: Document entity for metadata
            
        Returns:
            Processing service response data
        """
        last_exception = None
        
        for attempt_number in range(self.max_retries):
            try:
                self.logger.info(f"Calling processing service for document {document.id} (attempt {attempt_number + 1}/{self.max_retries})")
                
                timeout = aiohttp.ClientTimeout(total=self.timeout)

                # Get authentication headers
                headers = self._get_auth_headers()

                async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                    # Create form data for document upload
                    upload_form = aiohttp.FormData()
                    upload_form.add_field(
                        'file',
                        file_content,
                        filename=document.original_filename,
                        content_type=self._get_content_type(document.original_filename)
                    )
                    
                    async with session.post(
                        f"{self.api_base_url}/upload",
                        data=upload_form
                    ) as response:
                        self.logger.debug(f"Processing service response status: {response.status}")
                        
                        if response.status == 200:
                            response_data = await response.json()
                            self.logger.info(f"Successfully received response from processing service for document {document.id}")
                            return response_data
                        else:
                            error_text = await response.text()
                            raise DocumentProcessingError(
                                f"Processing service returned status {response.status}: {error_text}"
                            )
                            
            except asyncio.TimeoutError as e:
                last_exception = DocumentProcessingError(f"Processing service call timed out after {self.timeout} seconds")
                self.logger.warning(f"Processing service timeout for document {document.id} (attempt {attempt_number + 1})")
                
            except aiohttp.ClientError as e:
                last_exception = DocumentProcessingError(f"Processing service client error: {str(e)}")
                self.logger.warning(f"Processing service client error for document {document.id} (attempt {attempt_number + 1}): {str(e)}")
                
            except Exception as e:
                last_exception = DocumentProcessingError(f"Unexpected processing service error: {str(e)}")
                self.logger.error(f"Unexpected processing service error for document {document.id} (attempt {attempt_number + 1}): {str(e)}")
            
            # Wait before retry with exponential backoff
            if attempt_number < self.max_retries - 1:
                backoff_delay = 2 ** attempt_number
                self.logger.info(f"Waiting {backoff_delay} seconds before retry...")
                await asyncio.sleep(backoff_delay)
        
        # All retry attempts failed
        raise last_exception or DocumentProcessingError("All processing service retry attempts failed")

    async def process_document(self, document: Document) -> Tuple[List[DocumentChunk], str]:
        """
        Process a document using the remote document processing service.
        
        Args:
            document: Document entity to process
            
        Returns:
            Tuple of (document chunks, complete text)
        """
        try:
            self.logger.info(f"Starting remote document processing for document {document.id}")
            
            # Get file content from storage
            file_content = await self.storage_service.get_file_content(document.file_path)
            if file_content is None:
                raise DocumentProcessingError(f"Could not retrieve file content for document {document.id}")
            
            # Call processing service with retries
            processing_response = await self._call_processing_service_with_retry(file_content, document)
            
            # Parse response and convert to local domain objects
            document_chunks, extracted_text = self._parse_processing_service_response(processing_response, document)
            
            self.logger.info(f"Successfully processed document {document.id} remotely, created {len(document_chunks)} chunks")
            return document_chunks, extracted_text
            
        except Exception as e:
            self.logger.error(f"Remote document processing failed for {document.id}: {str(e)}")
            raise DocumentProcessingError(f"Remote document processing failed: {str(e)}")