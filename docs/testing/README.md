# Testing Guide

This document provides comprehensive information about testing in the Document Chat Service.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_pytest_collection.py

# Run tests matching a pattern
poetry run pytest -k "test_dependency"
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                          # Shared fixtures
├── test_dynamic_context_window.py       # Integration test
├── unit/
│   ├── __init__.py
│   ├── test_pytest_collection.py        # Pytest configuration tests
│   ├── api/
│   │   ├── __init__.py
│   │   ├── test_dependency_injection.py # DI configuration tests
│   │   └── test_route_method_compatibility.py # API route tests
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── test_entities.py             # Domain entity tests
│   │   ├── test_evaluation_entities.py  # Evaluation domain tests
│   │   ├── test_exceptions.py           # Exception tests
│   │   └── test_value_objects.py        # Value object tests
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── test_collection_management.py
│   │   ├── test_context_retrieval.py
│   │   └── test_document_management.py
│   └── test_use_cases/
│       ├── test_chat_interaction.py
│       ├── test_intent_classification.py
│       └── test_query_transformation.py
└── integration/
    ├── __init__.py
    ├── adapters/
    │   └── __init__.py
    └── api/
        └── __init__.py
```

## Running Tests

### Basic Commands

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run with very verbose output (show test names)
poetry run pytest -vv

# Run specific test file
poetry run pytest tests/unit/domain/test_entities.py

# Run specific test class
poetry run pytest tests/unit/api/test_dependency_injection.py::TestDocumentManagementUseCase

# Run specific test function
poetry run pytest tests/unit/test_pytest_collection.py::test_known_domain_classes_have_test_false

# Run tests matching pattern
poetry run pytest -k "dependency or injection"
```

### Coverage Reports

```bash
# Run with coverage report
poetry run pytest --cov=app

# Generate HTML coverage report
poetry run pytest --cov=app --cov-report=html

# Open coverage report (Windows)
start htmlcov/index.html

# Generate XML coverage report (for CI)
poetry run pytest --cov=app --cov-report=xml
```

### Markers

```bash
# Run only unit tests (fast)
poetry run pytest -m "not requires_ollama"

# Run integration tests that require Ollama
poetry run pytest -m "requires_ollama"

# List all available markers
poetry run pytest --markers
```

### Test Collection

```bash
# Show which tests would be run (don't execute)
poetry run pytest --collect-only

# Show test collection without warnings
poetry run pytest --collect-only -q
```

## Test Categories

### 1. Regression Prevention Tests

These tests prevent previously fixed issues from recurring.

#### Pytest Collection Tests (`test_pytest_collection.py`)
**Purpose**: Prevent pytest from incorrectly collecting domain classes as tests

**What it tests**:
- Domain classes starting with "Test" have `__test__ = False`
- Pytest collection runs without warnings
- Known problematic classes (TestCase, TestCaseLoadError) are properly marked

**Run**:
```bash
poetry run pytest tests/unit/test_pytest_collection.py -v
```

**Related Issue**: [docs/failures/003-pytest-collection-warnings.md](../failures/003-pytest-collection-warnings.md)

#### Dependency Injection Tests (`test_dependency_injection.py`)
**Purpose**: Ensure FastAPI dependency injection is correctly configured

**What it tests**:
- Dependency provider functions have correct signatures
- Use case constructors match dependency providers
- No duplicate dependency functions exist
- Aliases are properly configured

**Run**:
```bash
poetry run pytest tests/unit/api/test_dependency_injection.py -v
```

**Example failure scenario**:
```python
# This would fail the test
def get_document_management_use_case(
    storage_service: StorageService,  # Wrong parameter!
    ...
) -> DocumentManagementUseCase:
    return DocumentManagementUseCase(...)  # TypeError at runtime
```

#### Route Method Compatibility Tests (`test_route_method_compatibility.py`)
**Purpose**: Ensure API routes call methods that exist on use cases

**What it tests**:
- Routes call existing use case methods
- Method names match exactly
- Methods have correct signatures
- Methods are async (since routes await them)

**Run**:
```bash
poetry run pytest tests/unit/api/test_route_method_compatibility.py -v
```

**Example failure scenario**:
```python
# Route calls non-existent method
@router.get("/documents/{document_id}")
async def get_document(document_id: str, use_case: DocumentManagementUseCase):
    return await use_case.get_document_by_id(document_id)  # Method doesn't exist!
```

### 2. Domain Tests

Test business logic and domain entities.

```bash
# Run all domain tests
poetry run pytest tests/unit/domain/ -v

# Test specific domain area
poetry run pytest tests/unit/domain/test_entities.py -v
```

### 3. Use Case Tests

Test application use cases.

```bash
# Run all use case tests
poetry run pytest tests/unit/use_cases/ tests/unit/test_use_cases/ -v

# Test specific use case
poetry run pytest tests/unit/use_cases/test_document_management.py -v
```

### 4. Integration Tests

Test interactions between components.

```bash
# Run integration tests (may require services)
poetry run pytest tests/integration/ -v

# Skip tests requiring Ollama
poetry run pytest tests/integration/ -m "not requires_ollama"
```

## Writing Tests

### Using Fixtures

The project provides many reusable fixtures in `tests/conftest.py`:

```python
import pytest

def test_document_creation(sample_document, sample_collection):
    """Example using fixtures from conftest.py"""
    assert sample_document.collection_id == sample_collection.id
    assert sample_document.status == DocumentStatus.PROCESSED
```

### Mocking Services

```python
from unittest.mock import AsyncMock, Mock

def test_with_mocks(mock_embedding_service, mock_vector_store):
    """Example using mock services"""
    # Mock services are already configured with default behaviors
    use_case = DocumentManagementUseCase(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        ...
    )
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation(mock_document_repository):
    """Example async test"""
    document = await mock_document_repository.find_by_id("doc-123")
    assert document is not None
```

### Testing Exceptions

```python
import pytest
from app.core.domain.exceptions import DocumentNotFoundError

@pytest.mark.asyncio
async def test_raises_exception():
    """Example testing exceptions"""
    with pytest.raises(DocumentNotFoundError):
        await use_case.get_document("non-existent-id")
```

### Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("extension,expected_type", [
    ("pdf", DocumentType.PDF),
    ("docx", DocumentType.DOCX),
    ("md", DocumentType.MARKDOWN),
])
def test_document_type_detection(extension, expected_type):
    """Example parameterized test"""
    doc_type = determine_document_type(f"file.{extension}")
    assert doc_type == expected_type
```

## Continuous Integration

### Pre-commit Checks

Before committing, run:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=term-missing

# Check for collection warnings
poetry run pytest --collect-only -q
```

### CI Pipeline

The CI pipeline runs:
1. Pytest collection check (no warnings)
2. All unit tests (excluding Ollama-dependent tests)
3. Coverage report generation
4. Code quality checks

### Adding Tests to CI

Tests are automatically run on:
- Pull requests to `main`
- Pushes to `main`
- Manual workflow dispatch

## Troubleshooting

### Common Issues

#### Tests Can't Find Modules

**Problem**: `ModuleNotFoundError: No module named 'app'`

**Solution**:
```bash
# Make sure you're using poetry run
poetry run pytest

# Or activate the virtual environment
poetry shell
pytest
```

#### Pytest Collection Warnings

**Problem**: `PytestCollectionWarning: cannot collect test class 'TestCase'`

**Solution**: Add `__test__ = False` to the domain class:
```python
@dataclass
class TestCase:
    __test__ = False  # Tell pytest this is not a test class
    # ... rest of class
```

#### Async Tests Not Running

**Problem**: Tests hang or don't run

**Solution**: Make sure you have the `@pytest.mark.asyncio` decorator:
```python
@pytest.mark.asyncio
async def test_my_async_function():
    result = await my_async_function()
    assert result is not None
```

#### Fixtures Not Found

**Problem**: `fixture 'my_fixture' not found`

**Solution**: Check that:
1. Fixture is defined in `conftest.py` or imported test file
2. Fixture name matches exactly (no typos)
3. `conftest.py` is in the correct location

#### Tests Pass Locally But Fail in CI

**Common causes**:
1. Missing environment variables
2. Tests depend on local files/services
3. Different Python versions
4. Timezone differences

**Debug**:
```bash
# Run tests like CI does
poetry run pytest -m "not requires_ollama" --tb=short
```

### Getting Help

- Check [docs/failures/](../failures/) for documented issues
- Review test output with `-vv` for detailed information
- Use `--tb=short` for concise error traces
- Use `--pdb` to drop into debugger on failure

## Best Practices

1. **Keep tests fast**: Unit tests should run in milliseconds
2. **Use fixtures**: Don't repeat setup code
3. **Test one thing**: Each test should verify one behavior
4. **Clear names**: Test names should describe what they test
5. **Arrange-Act-Assert**: Structure tests clearly
6. **Mock external dependencies**: Don't make real API calls in unit tests
7. **Clean up**: Use fixtures for setup/teardown
8. **Document complex tests**: Add docstrings explaining why
9. **Run tests before committing**: Catch issues early
10. **Keep coverage high**: Aim for >80% code coverage

## Test Naming Conventions

```python
# Good test names
def test_document_creation_succeeds_with_valid_data()
def test_upload_raises_exception_when_file_too_large()
def test_list_documents_returns_empty_list_for_new_collection()

# Bad test names
def test_document()
def test_case_1()
def test_it_works()
```

## Example Test Template

```python
"""
Tests for [Component Name]

This module tests [brief description of what's being tested].
"""
import pytest
from unittest.mock import AsyncMock, Mock

from app.core.domain.entities.document import Document
from app.core.use_cases.document_management import DocumentManagementUseCase


class TestDocumentManagement:
    """Test suite for DocumentManagementUseCase"""

    @pytest.fixture
    def use_case(self, mock_document_repository, mock_vector_store):
        """Create use case instance with mocked dependencies"""
        return DocumentManagementUseCase(
            document_repo=mock_document_repository,
            vector_store=mock_vector_store,
            # ... other dependencies
        )

    @pytest.mark.asyncio
    async def test_get_document_succeeds_when_document_exists(
        self, use_case, sample_document
    ):
        """
        GIVEN a document exists in the repository
        WHEN get_document is called with valid ID
        THEN the document is returned
        """
        # Arrange
        use_case.document_repo.find_by_id.return_value = sample_document

        # Act
        result = await use_case.get_document(sample_document.id)

        # Assert
        assert result == sample_document
        use_case.document_repo.find_by_id.assert_called_once_with(
            sample_document.id
        )

    @pytest.mark.asyncio
    async def test_get_document_raises_exception_when_not_found(self, use_case):
        """
        GIVEN a document does not exist
        WHEN get_document is called
        THEN DocumentNotFoundError is raised
        """
        # Arrange
        use_case.document_repo.find_by_id.return_value = None

        # Act & Assert
        with pytest.raises(DocumentNotFoundError):
            await use_case.get_document("non-existent-id")
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [Python unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Project failure documentation](../failures/)
