# Testing Quick Reference

A quick reference for common testing commands and patterns.

## Common Commands

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific file
poetry run pytest tests/unit/test_pytest_collection.py

# Run specific test
poetry run pytest tests/unit/api/test_dependency_injection.py::TestDocumentManagementUseCase::test_get_document_management_use_case_is_alias

# Run tests matching pattern
poetry run pytest -k "dependency"

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Skip slow tests (Ollama)
poetry run pytest -m "not requires_ollama"

# Show which tests would run
poetry run pytest --collect-only
```

## Test Categories

| Category | Command | Purpose |
|----------|---------|---------|
| All tests | `pytest` | Run everything |
| Unit tests only | `pytest -m "not requires_ollama"` | Fast tests only |
| Domain tests | `pytest tests/unit/domain/` | Business logic |
| Use case tests | `pytest tests/unit/use_cases/` | Application layer |
| API tests | `pytest tests/unit/api/` | API configuration |
| Integration tests | `pytest tests/integration/` | Component interaction |

## Pre-commit Checklist

```bash
# 1. Run all fast tests
poetry run pytest -m "not requires_ollama"

# 2. Check for pytest collection warnings
poetry run pytest --collect-only -q

# 3. Run coverage check
poetry run pytest --cov=app --cov-report=term-missing

# 4. Run linting (if configured)
poetry run ruff check .
```

## Common Fixtures

From `tests/conftest.py`:

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_collection` | Collection | Test collection entity |
| `sample_document` | Document | Test document entity |
| `sample_chunks` | List[DocumentChunk] | Test document chunks |
| `sample_embeddings` | List[EmbeddingVector] | Test embedding vectors |
| `mock_embedding_service` | AsyncMock | Mock embedding service |
| `mock_llm_service` | AsyncMock | Mock LLM service |
| `mock_vector_store` | AsyncMock | Mock vector store |
| `mock_bm25_service` | AsyncMock | Mock BM25 service |
| `mock_document_repository` | AsyncMock | Mock document repo |
| `mock_collection_repository` | AsyncMock | Mock collection repo |

## Test Templates

### Basic Test
```python
def test_something():
    """Test description"""
    # Arrange
    input_data = "test"

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_value
```

### Async Test
```python
@pytest.mark.asyncio
async def test_async_operation(mock_repository):
    """Test async operation"""
    result = await use_case.do_something()
    assert result is not None
```

### Exception Test
```python
def test_raises_exception():
    """Test exception is raised"""
    with pytest.raises(CustomException):
        function_that_should_raise()
```

### Parameterized Test
```python
@pytest.mark.parametrize("input,expected", [
    ("pdf", DocumentType.PDF),
    ("docx", DocumentType.DOCX),
])
def test_with_parameters(input, expected):
    """Test with multiple inputs"""
    result = convert(input)
    assert result == expected
```

### Using Fixtures
```python
def test_with_fixtures(sample_document, mock_repository):
    """Test using fixtures"""
    mock_repository.save.return_value = sample_document
    result = service.save(sample_document)
    assert result == sample_document
```

## Debugging Tests

```bash
# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Show full error traces
pytest --tb=long

# Show short error traces
pytest --tb=short

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run failed tests first, then others
pytest --ff
```

## Coverage Targets

| Metric | Target | Command |
|--------|--------|---------|
| Overall coverage | >80% | `pytest --cov=app` |
| Domain layer | >90% | `pytest --cov=app/core/domain` |
| Use cases | >85% | `pytest --cov=app/core/use_cases` |
| API layer | >75% | `pytest --cov=app/api` |

## Test File Locations

```
tests/
├── unit/
│   ├── test_pytest_collection.py        # Pytest config validation
│   ├── api/
│   │   ├── test_dependency_injection.py # DI validation
│   │   └── test_route_method_compatibility.py # Route/use case compatibility
│   ├── domain/
│   │   ├── test_entities.py
│   │   ├── test_evaluation_entities.py
│   │   ├── test_exceptions.py
│   │   └── test_value_objects.py
│   └── use_cases/
│       ├── test_collection_management.py
│       ├── test_context_retrieval.py
│       └── test_document_management.py
└── integration/
    └── (future integration tests)
```

## Pytest Markers

| Marker | Usage | Description |
|--------|-------|-------------|
| `@pytest.mark.asyncio` | Async tests | Mark test as async |
| `@pytest.mark.parametrize` | Multiple inputs | Run test with different parameters |
| `@pytest.mark.skip` | Skip test | Skip this test |
| `@pytest.mark.skipif(condition)` | Conditional skip | Skip if condition is true |
| `@pytest.mark.requires_ollama` | External service | Test requires Ollama running |

## CI/CD

Tests run automatically on:
- Pull requests to `main`
- Pushes to `main`

CI runs:
```bash
pytest -m "not requires_ollama" --cov=app --cov-report=xml
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Use `poetry run pytest` |
| Pytest collection warnings | Add `__test__ = False` to domain classes |
| Async test hangs | Add `@pytest.mark.asyncio` |
| Fixture not found | Check `conftest.py` and imports |
| Tests pass locally, fail in CI | Run `pytest -m "not requires_ollama"` |

## Performance

| Test Type | Target Speed | Notes |
|-----------|--------------|-------|
| Unit tests | <0.1s each | Mock all external dependencies |
| Integration tests | <5s each | May require services |
| Full suite | <30s | Excluding Ollama tests |

## Best Practices Checklist

- [ ] Test names describe what they test
- [ ] One assertion per test (generally)
- [ ] Tests are independent (can run in any order)
- [ ] External dependencies are mocked
- [ ] Fixtures are used for setup
- [ ] Tests follow Arrange-Act-Assert pattern
- [ ] Async tests have `@pytest.mark.asyncio`
- [ ] Coverage is >80% for new code
- [ ] Tests run fast (<0.1s for unit tests)
- [ ] No hardcoded paths or credentials

## Quick Test Writing Workflow

1. **Create test file** matching pattern `test_*.py`
2. **Import dependencies**
   ```python
   import pytest
   from app.core.domain.entities import MyEntity
   ```
3. **Write test function** starting with `test_`
   ```python
   def test_my_feature():
       """Clear description"""
       pass
   ```
4. **Run test**
   ```bash
   poetry run pytest tests/unit/test_my_feature.py -v
   ```
5. **Check coverage**
   ```bash
   poetry run pytest tests/unit/test_my_feature.py --cov=app.core
   ```

## Useful pytest Options

| Option | Description | Example |
|--------|-------------|---------|
| `-v` | Verbose output | `pytest -v` |
| `-vv` | Very verbose | `pytest -vv` |
| `-s` | Show print statements | `pytest -s` |
| `-x` | Stop on first failure | `pytest -x` |
| `-k PATTERN` | Run tests matching pattern | `pytest -k "document"` |
| `--lf` | Run last failed | `pytest --lf` |
| `--ff` | Failed first | `pytest --ff` |
| `--collect-only` | Show tests without running | `pytest --collect-only` |
| `--tb=short` | Short traceback | `pytest --tb=short` |
| `--pdb` | Debug on failure | `pytest --pdb` |
| `--cov=PATH` | Coverage for path | `pytest --cov=app` |
| `--cov-report=html` | HTML coverage report | `pytest --cov-report=html` |
| `-m MARKER` | Run tests with marker | `pytest -m "not requires_ollama"` |

## Environment Setup

```bash
# First time setup
git clone <repo>
cd chat-with-your-documents
poetry install

# Before running tests
poetry shell  # or use 'poetry run pytest'

# Verify setup
poetry run pytest --collect-only -q
```

## Links

- [Full Testing Guide](README.md)
- [Failure Documentation](../failures/)
- [pytest Documentation](https://docs.pytest.org/)
