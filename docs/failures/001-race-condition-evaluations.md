# Failure: Race Condition in Concurrent Evaluation Submissions

**Date:** 2025-11-11
**Category:** Bug
**Severity:** High
**Status:** Resolved
**Commit:** 73f41e2

## What We Tried

Initial implementation of batch evaluation submission used incremental counting to track completion:
- Query DB for `already_evaluated` count at start of batch
- Add new batch size to that count
- Save updated `evaluated_count` to run

```python
# Incorrect approach
already_evaluated = len(await result_repo.find_results_by_run_id(run_id))
new_count = already_evaluated + len(batch_results)
run.evaluated_count = new_count
```

Assumption: Batches process sequentially, so incremental counting is safe.

## What Happened

When multiple batches submitted concurrently (via async processing), `evaluated_count` became incorrect:

**Example with 4 total questions:**
1. Batch 1 (3 answers) and Batch 2 (1 answer) start ~same time
2. Both query DB at start, see `already_evaluated=0`
3. Batch 2 finishes first: sets `evaluated_count = 0 + 1 = 1`
4. Batch 1 finishes second: sets `evaluated_count = 0 + 3 = 3` (**overwrites!**)
5. Result: 4 results saved in DB but `evaluated_count=3`
6. `is_completed` stays `False` forever (expects 4 but sees 3)

**Impact:** Evaluation runs never complete, UI stuck showing "in progress"

## Why It Failed

**Root cause:** Classic read-modify-write race condition
- Multiple async operations read same stale value
- Each calculates update independently
- Last write wins, losing intermediate updates

**Wrong assumption:** Batch processing would be sequential enough to avoid races

## What We Learned

1. **Never use incremental counters with concurrent operations**
   - Read-modify-write patterns fail with async/parallel execution
   - Even "fast" operations can overlap

2. **Source of truth matters**
   - Actual saved results in DB are the truth
   - Cached/computed counts are derived state (can be wrong)

3. **Re-query before critical updates**
   - Always get fresh data before calculating dependent values
   - DB query cost << incorrect business logic cost

## What We Did Instead

**Fix:** Re-query actual results from DB before saving counts

```python
# Correct approach
all_results = await result_repo.find_results_by_run_id(run_id)
correct_count = sum(1 for r in all_results if r.is_correct)
incorrect_count = len(all_results) - correct_count

run.evaluated_count = len(all_results)
run.correct_count = correct_count
run.incorrect_count = incorrect_count
```

**Why this works:**
- Queries actual saved results (source of truth)
- Counts correct/incorrect from actual data
- No race condition: DB serializes concurrent writes
- Handles any number of concurrent batches correctly

## Prevent Future Occurrences

**Checklist for async operations:**
- [ ] Identify shared state being modified
- [ ] Check for read-modify-write patterns
- [ ] If found, either:
  - Use atomic DB operations (INCREMENT, etc.)
  - Re-query source of truth before updates
  - Use proper locking/transaction isolation
- [ ] Test with concurrent requests (not just sequential)

**Code patterns to avoid:**
```python
# BAD: Incremental counting with concurrent operations
count = await get_current_count()
new_count = count + increment
await save_count(new_count)

# GOOD: Re-query source of truth
all_items = await get_all_items()
count = len(all_items)
await save_count(count)
```

## Related Issues

- File: `app/core/use_cases/evaluation/submit_plugin_evaluation.py:24`
- Testing gap: No concurrent submission tests existed
- Future: Add `test_concurrent_batch_submission()` to prevent regression
