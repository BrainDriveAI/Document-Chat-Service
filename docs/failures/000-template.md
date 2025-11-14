# Failure: [Brief Title]

**Date:** YYYY-MM-DD
**Category:** [Performance | Bug | Integration | Design | Architecture | Assumption]
**Severity:** [Low | Medium | High | Critical]
**Status:** [Investigating | Root Cause Found | Resolved | Workaround Applied]

## What We Tried

Detailed description of the approach/experiment that failed.

**Code example (if applicable):**
```python
# The approach that didn't work
```

**Assumptions made:**
- Assumption 1
- Assumption 2

## What Happened

What actually occurred? Include:
- Error messages
- Unexpected behavior
- Metrics/measurements
- Impact on system

**Error output:**
```
Paste error message here
```

## Why It Failed

**Root cause analysis:**
- What assumption was wrong?
- What was misunderstood?
- What edge case was missed?

**Technical explanation:**
Detailed explanation of why this approach couldn't work.

## What We Learned

**Key insights:**
1. Insight 1
2. Insight 2
3. Insight 3

**Patterns to avoid:**
- Anti-pattern 1
- Anti-pattern 2

## What We Did Instead

**Correct approach:**
```python
# The solution that actually worked
```

**Why this works:**
Explanation of why the correct approach solves the problem.

**Tradeoffs:**
- Pro: ...
- Con: ...

## Prevent Future Occurrences

**Checklist for similar situations:**
- [ ] Check item 1
- [ ] Check item 2
- [ ] Verify assumption 3

**Code patterns to avoid:**
```python
# BAD: Don't do this
bad_example()

# GOOD: Do this instead
good_example()
```

**Testing strategy:**
- Test case 1 to prevent regression
- Test case 2 to catch edge cases

## Related Documentation

- **ADRs:** Link to related architecture decisions
- **Files affected:** `app/path/to/file.py:123`
- **Commits:** Link to fix commit
- **Issues:** Link to GitHub issue
- **Related failures:** Link to similar failures
