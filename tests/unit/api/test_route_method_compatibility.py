"""
Test API route and use case method compatibility.

This test ensures that API route handlers call methods that actually exist
on the use case classes they depend on, with the correct signatures.

This prevents runtime errors like:
- AttributeError: 'UseCase' object has no attribute 'method_name'
- TypeError: method() got an unexpected keyword argument
"""
import inspect
import pytest
from typing import get_type_hints
from unittest.mock import Mock, AsyncMock, patch


class TestDocumentRouteMethodCompatibility:
    """Test document API routes call correct use case methods."""

    def test_list_documents_route_uses_correct_method(self):
        """Verify list_documents route calls existing use case method."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        # The route should call list_documents_by_collection (not list_documents)
        assert hasattr(DocumentManagementUseCase, "list_documents_by_collection"), (
            "DocumentManagementUseCase missing 'list_documents_by_collection' method. "
            "API route expects this method to exist."
        )

        # Verify it does NOT have a method named just "list_documents"
        # (to prevent confusion)
        if hasattr(DocumentManagementUseCase, "list_documents"):
            method = getattr(DocumentManagementUseCase, "list_documents")
            # If it exists, it should be an alias to list_documents_by_collection
            assert method is DocumentManagementUseCase.list_documents_by_collection, (
                "If 'list_documents' exists, it should be an alias to "
                "'list_documents_by_collection'"
            )

    def test_get_document_route_uses_correct_method(self):
        """Verify get_document route calls existing use case method."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        # The route should call get_document (not get_document_by_id)
        assert hasattr(DocumentManagementUseCase, "get_document"), (
            "DocumentManagementUseCase missing 'get_document' method. "
            "API route expects this method to exist."
        )

        # Verify method signature accepts document_id parameter
        sig = inspect.signature(DocumentManagementUseCase.get_document)
        params = list(sig.parameters.keys())

        assert "document_id" in params, (
            f"get_document method should accept 'document_id' parameter. "
            f"Got parameters: {params}"
        )

    def test_delete_document_route_uses_correct_method(self):
        """Verify delete_document route calls existing use case method."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        assert hasattr(DocumentManagementUseCase, "delete_document"), (
            "DocumentManagementUseCase missing 'delete_document' method. "
            "API route expects this method to exist."
        )

        # Verify method signature
        sig = inspect.signature(DocumentManagementUseCase.delete_document)
        params = list(sig.parameters.keys())

        assert "document_id" in params, (
            f"delete_document method should accept 'document_id' parameter. "
            f"Got parameters: {params}"
        )

    def test_process_document_route_uses_correct_method(self):
        """Verify process_document background task calls existing method."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        assert hasattr(DocumentManagementUseCase, "process_document"), (
            "DocumentManagementUseCase missing 'process_document' method. "
            "Background task expects this method to exist."
        )


class TestRouteHandlerSignatures:
    """Verify route handlers have correct signatures."""

    def test_list_documents_route_handler(self):
        """Verify list_documents route handler implementation."""
        # Read the route file to check the implementation
        import ast
        from pathlib import Path

        route_file = Path("app/api/routes/documents.py")
        with open(route_file, "r", encoding="utf-8") as f:
            source = f.read()

        # Parse the source
        tree = ast.parse(source)

        # Find the list_documents function
        list_docs_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "list_documents":
                list_docs_func = node
                break

        assert list_docs_func is not None, "list_documents route function not found"

        # Check that it calls the correct method
        # Look for method calls in the function body
        method_calls = []
        for node in ast.walk(list_docs_func):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "document_use_case":
                    method_calls.append(node.attr)

        assert "list_documents_by_collection" in method_calls, (
            f"list_documents route should call "
            f"'document_use_case.list_documents_by_collection()'. "
            f"Found calls: {method_calls}"
        )

    def test_get_document_route_handler(self):
        """Verify get_document route handler implementation."""
        import ast
        from pathlib import Path

        route_file = Path("app/api/routes/documents.py")
        with open(route_file, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        # Find the get_document function
        get_doc_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_document":
                get_doc_func = node
                break

        assert get_doc_func is not None, "get_document route function not found"

        # Check that it calls the correct method
        method_calls = []
        for node in ast.walk(get_doc_func):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "document_use_case":
                    method_calls.append(node.attr)

        # Should call get_document, not get_document_by_id
        assert "get_document" in method_calls, (
            f"get_document route should call 'document_use_case.get_document()'. "
            f"Found calls: {method_calls}"
        )

        # Should NOT call get_document_by_id
        assert "get_document_by_id" not in method_calls, (
            f"get_document route should NOT call 'get_document_by_id'. "
            f"Use 'get_document' instead."
        )


class TestUseCaseMethodExistence:
    """Comprehensive test of all use case methods used by routes."""

    def test_all_document_management_methods_exist(self):
        """Verify all methods called by document routes exist on use case."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        # Methods that MUST exist (called by routes)
        required_methods = {
            "get_document": "Get document by ID",
            "list_documents_by_collection": "List documents in collection",
            "delete_document": "Delete document",
            "process_document": "Process document (background task)",
        }

        missing_methods = []
        for method_name, description in required_methods.items():
            if not hasattr(DocumentManagementUseCase, method_name):
                missing_methods.append(f"{method_name} ({description})")

        assert not missing_methods, (
            f"DocumentManagementUseCase missing required methods:\n"
            f"{chr(10).join('  - ' + m for m in missing_methods)}\n\n"
            f"These methods are called by API routes and must exist."
        )

    def test_all_document_management_methods_are_async(self):
        """Verify use case methods are async (since routes await them)."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        methods_to_check = [
            "get_document",
            "list_documents_by_collection",
            "delete_document",
            "process_document",
        ]

        non_async_methods = []
        for method_name in methods_to_check:
            if hasattr(DocumentManagementUseCase, method_name):
                method = getattr(DocumentManagementUseCase, method_name)
                if not inspect.iscoroutinefunction(method):
                    non_async_methods.append(method_name)

        assert not non_async_methods, (
            f"The following methods should be async but are not:\n"
            f"{chr(10).join('  - ' + m for m in non_async_methods)}\n\n"
            f"API routes await these methods, so they must be async."
        )


class TestCollectionRouteMethodCompatibility:
    """Test collection API routes call correct use case methods."""

    def test_collection_use_case_methods_exist(self):
        """Verify collection use case has required methods."""
        from app.core.use_cases.collection_management import CollectionManagementUseCase

        required_methods = [
            "create_collection",
            "get_collection",
            "list_collections",
            "update_collection",
            "delete_collection",
        ]

        for method_name in required_methods:
            assert hasattr(CollectionManagementUseCase, method_name), (
                f"CollectionManagementUseCase missing '{method_name}' method"
            )
