# Failure: Ollama-Dependent Tests Failing in CI

**Date:** 2025-11-13
**Category:** Integration
**Severity:** Medium
**Status:** Resolved
**Commit:** 7320dce

## What We Tried

Comprehensive test suite for dynamic context window detection included tests that call Ollama API:
- `test_get_model_info_full()` - Fetches model info from Ollama
- `test_list_all_local_models()` - Lists all local models
- `test_refresh_cache()` - Refreshes model info cache

**Assumption:** CI environment would have Ollama running (like local dev)

## What Happened

GitHub Actions CI runs failed with 3 test failures:
- Tests attempted to connect to `http://localhost:11434`
- No Ollama instance running in CI container
- Connection refused errors

**Impact:**
- CI builds failed despite code being correct
- False negatives blocking PRs
- 223 other tests passed, but 3 failures blocked merge

## Why It Failed

**Root cause:** Environment mismatch between local dev and CI

**Local development:**
- Ollama runs on host machine
- Docker Compose mounts Ollama socket
- Tests have real Ollama instance available

**CI environment (GitHub Actions):**
- Ubuntu runner with no Ollama installation
- No external services available by default
- Would need to install/run Ollama in workflow (slow, complex)

**Wrong assumption:** CI would mirror local dev environment

## What We Learned

1. **Integration tests need environment markers**
   - Mark tests requiring external services
   - Make CI skip them unless explicitly enabled
   - Keep unit tests separate from integration tests

2. **CI != Local development**
   - CI is minimal: only what's in repo + workflow config
   - External services (databases, APIs, LLMs) not available by default
   - Must explicitly choose to add them (tradeoff: speed vs coverage)

3. **Fast CI > Complete CI for this project**
   - Ollama tests validate integration, not core logic
   - Core logic tested in 223 other tests
   - Better to have fast CI, manual integration testing

## What We Did Instead

**Fix:** Mark Ollama-dependent tests and skip in CI

1. Added pytest marker to tests requiring Ollama:
```python
@pytest.mark.requires_ollama
async def test_get_model_info_full(ollama_service):
    # Test that needs real Ollama instance
    ...
```

2. Updated CI workflow to skip marked tests:
```yaml
# .github/workflows/tests.yml
- name: Run tests
  run: poetry run pytest -m "not requires_ollama"
```

**Result:**
- CI runs: 223 passed, 3 skipped, 4 deselected
- Local dev: 226 passed (all tests, including Ollama ones)
- Fast CI builds (~2 min instead of ~10+ min if installing Ollama)

## Prevent Future Occurrences

**Checklist for new tests:**
- [ ] Does this test call external API/service?
- [ ] If yes, mark with `@pytest.mark.requires_<service>`
- [ ] Document in README which markers exist
- [ ] Test locally with markers: `pytest -m "not requires_ollama"`

**Common markers to use:**
```python
@pytest.mark.requires_ollama      # Needs Ollama running
@pytest.mark.requires_postgres    # Needs PostgreSQL (future)
@pytest.mark.requires_redis       # Needs Redis (future)
@pytest.mark.slow                 # Long-running tests (>5s)
```

**pytest.ini configuration:**
```ini
[tool:pytest]
markers =
    requires_ollama: Tests requiring Ollama instance
    requires_postgres: Tests requiring PostgreSQL
    requires_redis: Tests requiring Redis
    slow: Tests that take >5 seconds
```

## Alternatives Considered

**Option 1: Install Ollama in CI**
- Pro: Complete test coverage in CI
- Con: Slow CI (5-10 min to install Ollama)
- Con: Resource-heavy (model downloads)
- Con: Flaky (external dependency)
- **Rejected:** Speed more important than integration coverage for this project

**Option 2: Mock Ollama responses**
- Pro: Fast, no external dependencies
- Con: Not testing real integration
- Con: Mocks can drift from reality
- **Rejected:** Defeats purpose of integration tests

**Option 3: Separate integration test job (chosen for markers)**
- Pro: Can run integration tests on-demand
- Pro: Fast default CI
- **Chosen:** Implemented via pytest markers

## Related Issues

- File: `tests/test_dynamic_context_window.py`
- File: `.github/workflows/tests.yml`
- Future: Consider scheduled integration test runs (nightly) with real Ollama
- Testing philosophy: Fast unit tests in CI, manual integration tests locally
