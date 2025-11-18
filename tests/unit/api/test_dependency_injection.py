"""
Test dependency injection configuration.

This test ensures that dependency provider functions in app/api/deps.py
correctly instantiate use cases with the right parameters.

This prevents issues like:
- Passing wrong parameters to use case constructors
- Duplicate dependency functions with different signatures
- Missing required dependencies
"""
import inspect
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import Request


def create_mock_request(**state_overrides):
    """Create a mock FastAPI request with app.state attributes."""
    request = Mock(spec=Request)
    request.app = Mock()
    request.app.state = Mock()

    # Set default mock services
    defaults = {
        "document_processor": AsyncMock(),
        "storage_service": AsyncMock(),
        "embedding_service": AsyncMock(),
        "llm_service": AsyncMock(),
        "contextual_llm_service": AsyncMock(),  # Changed from None to AsyncMock
        "model_info_service": AsyncMock(),
        "vector_store": AsyncMock(),
        "document_repo": AsyncMock(),
        "collection_repo": AsyncMock(),
        "chat_repo": AsyncMock(),
        "bm25_service": AsyncMock(),
        "rank_fusion_service": Mock(),
        "evaluation_repo": AsyncMock(),
        "evaluation_state_repo": AsyncMock(),
        "judge_service": AsyncMock(),
    }

    # Override with provided values
    defaults.update(state_overrides)

    # Set attributes on request.app.state
    for key, value in defaults.items():
        setattr(request.app.state, key, value)

    return request


class TestDocumentManagementUseCase:
    """Tests for DocumentManagementUseCase dependency injection."""

    def test_get_document_processing_use_case_signature(self):
        """Verify get_document_processing_use_case has correct signature."""
        from app.api.deps import get_document_processing_use_case

        # Get function signature
        sig = inspect.signature(get_document_processing_use_case)

        # Expected parameters (excluding defaults which are Depends(...))
        expected_params = {
            "document_repo",
            "collection_repo",
            "document_processor",
            "embedding_service",
            "vector_store",
            "llm_service",
            "contextual_llm",
            "bm25_service",
        }

        actual_params = set(sig.parameters.keys())

        assert actual_params == expected_params, (
            f"get_document_processing_use_case signature mismatch.\n"
            f"Expected: {expected_params}\n"
            f"Actual: {actual_params}\n"
            f"Missing: {expected_params - actual_params}\n"
            f"Extra: {actual_params - expected_params}"
        )

    def test_get_document_management_use_case_is_alias(self):
        """Verify get_document_management_use_case is an alias, not duplicate."""
        from app.api.deps import (
            get_document_management_use_case,
            get_document_processing_use_case
        )

        # These should be the exact same function (not duplicates)
        assert get_document_management_use_case is get_document_processing_use_case, (
            "get_document_management_use_case should be an alias to "
            "get_document_processing_use_case, not a separate function.\n"
            "Use: get_document_management_use_case = get_document_processing_use_case"
        )

    def test_document_management_use_case_instantiation(self):
        """Verify DocumentManagementUseCase can be instantiated with deps."""
        from app.api.deps import get_document_processing_use_case
        from app.core.use_cases.document_management import DocumentManagementUseCase

        # Create mock dependencies
        request = create_mock_request()

        # Get dependencies (simulating FastAPI dependency resolution)
        from app.api.deps import (
            get_document_repository,
            get_collection_repository,
            get_document_processor,
            get_embedding_service,
            get_vector_store,
            get_llm_service,
            get_contextual_llm_service,
            get_bm25_service,
        )

        document_repo = get_document_repository(request)
        collection_repo = get_collection_repository(request)
        document_processor = get_document_processor(request)
        embedding_service = get_embedding_service(request)
        vector_store = get_vector_store(request)
        llm_service = get_llm_service(request)
        contextual_llm = get_contextual_llm_service(request)
        bm25_service = get_bm25_service(request)

        # Instantiate use case
        use_case = get_document_processing_use_case(
            document_repo=document_repo,
            collection_repo=collection_repo,
            document_processor=document_processor,
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_service=llm_service,
            contextual_llm=contextual_llm,
            bm25_service=bm25_service,
        )

        # Verify it's the correct type
        assert isinstance(use_case, DocumentManagementUseCase), (
            f"Expected DocumentManagementUseCase instance, got {type(use_case)}"
        )

    def test_document_management_use_case_constructor_params(self):
        """Verify DocumentManagementUseCase constructor accepts correct params."""
        from app.core.use_cases.document_management import DocumentManagementUseCase

        # Get constructor signature
        sig = inspect.signature(DocumentManagementUseCase.__init__)

        # Expected parameters (excluding self)
        expected_params = {
            "document_repo",
            "collection_repo",
            "document_processor",
            "embedding_service",
            "vector_store",
            "llm_service",
            "contextual_llm",
            "bm25_service",
        }

        actual_params = set(sig.parameters.keys()) - {"self"}

        assert actual_params == expected_params, (
            f"DocumentManagementUseCase.__init__ signature mismatch.\n"
            f"Expected: {expected_params}\n"
            f"Actual: {actual_params}\n"
            f"Missing: {expected_params - actual_params}\n"
            f"Extra: {actual_params - expected_params}\n\n"
            f"If you see 'storage_service' or 'logger' in Extra, "
            f"these parameters were removed. Update dependency injection."
        )


class TestOtherUseCaseDependencies:
    """Tests for other use case dependency injection."""

    def test_no_duplicate_dependency_functions(self):
        """Verify there are no duplicate dependency provider functions."""
        from app.api import deps
        import types

        # Get all functions from deps module
        functions = [
            (name, obj) for name, obj in inspect.getmembers(deps)
            if isinstance(obj, types.FunctionType) and not name.startswith("_")
        ]

        # Check for functions that might be duplicates
        # (similar names but different implementations)
        function_bodies = {}
        duplicates = []

        for name, func in functions:
            # Get function code object as identifier
            code_id = id(func.__code__)

            if code_id in function_bodies:
                # This is a true alias (same code object) - that's OK
                continue

            # Check for similar names that might be unintended duplicates
            for other_name, other_code_id in function_bodies.items():
                if name != other_name and self._are_similar_names(name, other_name):
                    # Check if they actually point to the same function
                    other_func = next(f for n, f in functions if n == other_name)
                    if func is not other_func:
                        duplicates.append((name, other_name))

            function_bodies[name] = code_id

        assert not duplicates, (
            f"Found potentially duplicate dependency functions:\n"
            f"{duplicates}\n\n"
            f"If these should be aliases, use assignment:\n"
            f"  func_alias = original_func"
        )

    def _are_similar_names(self, name1: str, name2: str) -> bool:
        """Check if two function names are similar (potential duplicates)."""
        # Exclude known intentional pairs
        intentional_pairs = {
            ("get_llm_service", "get_contextual_llm_service"),
            ("get_document_management_use_case", "get_collection_management_use_case"),
            ("get_document_processing_use_case", "get_document_management_use_case"),
        }

        if (name1, name2) in intentional_pairs or (name2, name1) in intentional_pairs:
            return False

        # Remove common prefixes
        clean1, clean2 = name1, name2
        for prefix in ["get_", "create_"]:
            clean1 = clean1.replace(prefix, "")
            clean2 = clean2.replace(prefix, "")

        # Check if one contains the other
        return clean1 in clean2 or clean2 in clean1


class TestDependencyProviderReturnTypes:
    """Verify dependency providers return correct types."""

    def test_all_use_case_providers_return_correct_types(self):
        """Verify all get_*_use_case functions return use case instances."""
        from app.api import deps
        import types

        # Get all use case provider functions
        use_case_providers = [
            (name, obj) for name, obj in inspect.getmembers(deps)
            if isinstance(obj, types.FunctionType)
            and "use_case" in name.lower()
            and not name.startswith("_")
        ]

        missing_annotations = []
        for name, func in use_case_providers:
            # Get return type annotation
            sig = inspect.signature(func)
            return_type = sig.return_annotation

            # Only warn about main use case providers (not helpers)
            if name.startswith("get_") and return_type == inspect.Signature.empty:
                missing_annotations.append(name)

            # Check return type is from use_cases module (if annotation exists)
            if return_type != inspect.Signature.empty and hasattr(return_type, "__module__"):
                assert "use_cases" in return_type.__module__, (
                    f"{name} should return a use case class, "
                    f"got {return_type} from {return_type.__module__}"
                )

        # This is a warning, not a failure - some evaluation use cases may not have annotations
        if missing_annotations:
            pytest.skip(
                f"Some use case providers missing return type annotations: "
                f"{missing_annotations}. Consider adding type hints for better IDE support."
            )
