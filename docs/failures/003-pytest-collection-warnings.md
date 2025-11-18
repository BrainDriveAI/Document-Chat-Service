# Failure: Pytest Collection Warnings for Domain Classes

**Date:** 2025-11-18
**Status:** Resolved
**Impact:** Low (cosmetic warnings, tests still run)

## What Happened

Pytest was generating collection warnings when running `pytest --collect-only`:

```
app\core\domain\entities\evaluation.py:15: PytestCollectionWarning: cannot collect test class 'TestCase' because it has a __init__ constructor

app\core\domain\exceptions.py:105: PytestCollectionWarning: cannot collect test class 'TestCaseLoadError' because it has a __init__ constructor
```

## Root Cause

Pytest's default test discovery tries to collect any class whose name starts with "Test" as a test class. Two domain classes in the codebase matched this pattern:

1. **`TestCase`** (evaluation.py:16) - Domain entity representing evaluation test cases
2. **`TestCaseLoadError`** (exceptions.py:105) - Exception for test case loading failures

These are NOT test classes - they are domain entities and exceptions used in production code. However, pytest attempted to collect them because of the naming convention.

## Impact

- **Severity:** Low - cosmetic issue only
- **Tests:** All 230 tests collected and ran successfully despite warnings
- **User Impact:** No functional impact, just noisy output

## Fix

Added `__test__ = False` class attribute to both classes to explicitly tell pytest not to collect them:

```python
# app/core/domain/entities/evaluation.py
@dataclass
class TestCase:
    """Domain entity representing a test case for evaluation"""
    __test__ = False  # Tell pytest this is not a test class
    # ... rest of class

# app/core/domain/exceptions.py
class TestCaseLoadError(DomainException):
    """Raised when loading test cases from file fails"""
    __test__ = False  # Tell pytest this is not a test class
    pass
```

## Verification

```bash
# Before fix - had warnings
poetry run pytest --collect-only
# ============================== warnings summary ===============================
# app\core\domain\entities\evaluation.py:15: PytestCollectionWarning...

# After fix - clean output
poetry run pytest --collect-only
# ======================== 230 tests collected in 0.46s =========================

# All tests still pass
poetry run pytest -m "not requires_ollama"
# ================ 223 passed, 3 skipped, 4 deselected in 3.58s =================
```

## Lessons Learned

1. **Naming Convention Collision:** Be careful when naming domain classes that start with "Test" - pytest will try to collect them
2. **Alternative Solutions:**
   - Could rename classes (e.g., `EvaluationTestCase` instead of `TestCase`)
   - Could use `__test__ = False` attribute (chosen for minimal disruption)
   - Could configure pytest to ignore specific files/patterns
3. **Best Practice:** For domain entities representing test cases, consider names like `EvaluationCase`, `EvalTestCase`, or `QuestionCase` to avoid pytest conflicts

## Prevention

- **Code Review:** Check for domain class names starting with "Test"
- **CI Check:** Run `pytest --collect-only` in CI to catch collection warnings
- **Documentation:** Added this failure log to compound knowledge base

## Related Files

- `app/core/domain/entities/evaluation.py:16` - TestCase entity
- `app/core/domain/exceptions.py:105` - TestCaseLoadError exception
- All evaluation tests that use these classes remain unchanged
