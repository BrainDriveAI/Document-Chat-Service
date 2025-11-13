# Backend API Requirements: Evaluation State Persistence

## Context & Why We Need This

### Current Situation
The BrainDrive Chat With Docs Plugin currently manages evaluation orchestration **client-side** (in the browser):

**Current Evaluation Flow:**
1. User starts evaluation → Plugin calls `/api/evaluation/plugin/start-with-questions`
2. Backend returns `evaluation_run_id` + test questions with pre-fetched context
3. **Plugin orchestrates the rest:**
   - Generates LLM answers via BrainDrive backend (using user auth tokens)
   - Submits batches (3 questions) to backend for judging via `/api/evaluation/plugin/submit-with-questions`
   - Polls `/api/evaluation/results/{run_id}` until judging completes
   - Repeats for all questions
4. **Progress tracking:** Plugin saves state to browser `localStorage` after each batch
5. **Resume capability:** If interrupted, plugin loads from localStorage and continues

### The Problem
Current persistence uses browser `localStorage` with **1-hour timeout**:
- ❌ If user closes browser for > 1 hour → evaluation state **lost forever**
- ❌ If user switches devices → can't resume (localStorage is per-browser)
- ❌ If browser storage is cleared → evaluation state lost
- ❌ Browser storage limits (~5-10MB) could be exceeded with large evaluations

### Why Not Move Orchestration to Backend?
We considered having the backend run evaluations independently, but there's a critical blocker:

**Authentication Challenge:**
- Plugin uses **user auth tokens** to call BrainDrive backend
- BrainDrive uses **token rotation** (each refresh invalidates old token)
- If both plugin AND backend refresh tokens → race condition → one fails
- Backend can't reliably maintain user tokens without complex infrastructure

**Decision:** Keep orchestration in plugin (user must stay on page), but add **backend state persistence** for reliable resume capability.

---

## Solution: Backend State Persistence

### User Experience Goal
- ✅ User starts evaluation, accidentally closes browser
- ✅ Returns 2 days later, opens plugin
- ✅ Sees "Resume evaluation: 35/50 questions remaining"
- ✅ Clicks Resume → continues from where left off
- ✅ Works across devices (start on laptop, resume on desktop)

### Technical Approach
- Plugin saves evaluation state to **both** localStorage (fast) AND backend (durable)
- On resume: Plugin loads from backend first, falls back to localStorage if backend unavailable
- Backend retains state for **7 days** (configurable)
- No user tokens stored (security) - plugin uses its own current tokens

---

## Required API Endpoints

### 1. Save Evaluation State
**Endpoint:** `POST /api/evaluation/state/{evaluation_run_id}`

**Purpose:** Persist in-progress evaluation state for resume capability

**Authentication:** User JWT token (standard)

**Request Body:**
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": {
    "id": "ollama-qwen",
    "provider": "ollama",
    "name": "qwen3:8b",
    "isStreaming": false
  },
  "llm_model": "qwen3:8b",
  "persona": {
    "id": "persona-uuid",
    "name": "Helpful Assistant",
    "system_prompt": "You are a helpful assistant..."
  },
  "collection_id": "collection-uuid-123",
  "test_cases": [
    {
      "test_case_id": "tc-001",
      "question": "What is machine learning?",
      "expected_answer": "Machine learning is...",
      "retrieved_context": "Context: ML is a subset of AI...",
      "metadata": {}
    }
  ],
  "processed_question_ids": ["tc-001", "tc-002"],
  "current_batch": [],
  "last_updated": "2025-01-15T14:30:00Z"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "state_id": "state-uuid-789",
  "message": "Evaluation state saved successfully"
}
```

**Backend Logic:**
1. Validate `evaluation_run_id` exists in `evaluation_runs` table
2. Extract `user_id` from JWT token
3. **Upsert** state record (update if exists, insert if new):
   - Key: `(evaluation_run_id, user_id)` (composite unique constraint)
   - Store full request body as JSON
   - Update `last_updated` timestamp
4. Set automatic cleanup: Delete records older than 7 days

**Error Responses:**
- `401 Unauthorized` - Invalid/expired token
- `404 Not Found` - `evaluation_run_id` doesn't exist
- `400 Bad Request` - Invalid JSON schema

---

### 2. Load Evaluation State
**Endpoint:** `GET /api/evaluation/state/{evaluation_run_id}`

**Purpose:** Retrieve saved state for resume

**Authentication:** User JWT token

**Response:** `200 OK`
```json
{
  "state": {
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "model": { ... },
    "llm_model": "qwen3:8b",
    "persona": { ... },
    "collection_id": "collection-uuid-123",
    "test_cases": [ ... ],
    "processed_question_ids": ["tc-001", "tc-002"],
    "current_batch": [],
    "last_updated": "2025-01-15T14:30:00Z"
  },
  "metadata": {
    "age_hours": 2.5,
    "age_days": 0.1,
    "is_expired": false,
    "will_expire_in_hours": 165.5,
    "backend_evaluation_status": "running"
  }
}
```

**Backend Logic:**
1. Extract `user_id` from JWT token
2. Query: `SELECT * FROM evaluation_states WHERE evaluation_run_id = ? AND user_id = ?`
3. If not found or > 7 days old → Return `404`
4. Calculate metadata:
   - `age_hours` = `(now - last_updated) / 3600`
   - `age_days` = `age_hours / 24`
   - `is_expired` = `age_days > 7`
   - `will_expire_in_hours` = `(7 * 24) - age_hours`
5. Fetch backend evaluation status from `evaluation_runs` table
6. Return state + metadata

**Error Responses:**
- `401 Unauthorized` - Invalid/expired token
- `404 Not Found` - State not found or expired (> 7 days)

---

### 3. Delete Evaluation State
**Endpoint:** `DELETE /api/evaluation/state/{evaluation_run_id}`

**Purpose:** Cleanup when evaluation completes or user dismisses

**Authentication:** User JWT token

**Response:** `204 No Content`

**Backend Logic:**
1. Extract `user_id` from JWT token
2. Delete: `DELETE FROM evaluation_states WHERE evaluation_run_id = ? AND user_id = ?`
3. Return 204 even if record doesn't exist (idempotent)

**Error Responses:**
- `401 Unauthorized` - Invalid/expired token

---

### 4. List In-Progress Evaluations
**Endpoint:** `GET /api/evaluation/state/in-progress`

**Purpose:** Show all resumable evaluations for current user

**Authentication:** User JWT token

**Query Parameters:**
- `include_expired` (optional, default: `false`) - Include states > 7 days old

**Response:** `200 OK`
```json
{
  "evaluations": [
    {
      "run_id": "550e8400-e29b-41d4-a716-446655440000",
      "model_name": "qwen3:8b",
      "collection_id": "collection-uuid-123",
      "total_questions": 50,
      "processed_questions": 15,
      "remaining_questions": 35,
      "last_updated": "2025-01-15T14:30:00Z",
      "age_hours": 2.5,
      "age_days": 0.1,
      "is_expired": false,
      "progress_percentage": 30.0
    }
  ],
  "total_count": 1
}
```

**Backend Logic:**
1. Extract `user_id` from JWT token
2. Query: `SELECT * FROM evaluation_states WHERE user_id = ? ORDER BY last_updated DESC`
3. If `include_expired=false`: Filter out records > 7 days old
4. For each record:
   - Calculate summary stats
   - `progress_percentage` = `(processed_questions / total_questions) * 100`
5. Return array

**Error Responses:**
- `401 Unauthorized` - Invalid/expired token

---

## Database Schema

### New Table: `evaluation_states`

```sql
CREATE TABLE evaluation_states (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  evaluation_run_id UUID NOT NULL,
  user_id UUID NOT NULL,
  state_data JSONB NOT NULL,
  last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),

  -- Ensure one state per evaluation per user
  UNIQUE(evaluation_run_id, user_id),

  -- Foreign key to evaluation_runs table
  FOREIGN KEY (evaluation_run_id) REFERENCES evaluation_runs(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX idx_evaluation_states_run_id ON evaluation_states(evaluation_run_id);
CREATE INDEX idx_evaluation_states_user_id ON evaluation_states(user_id);
CREATE INDEX idx_evaluation_states_last_updated ON evaluation_states(last_updated);
CREATE INDEX idx_evaluation_states_user_updated ON evaluation_states(user_id, last_updated DESC);
```

**Schema Notes:**
- `state_data` is JSONB (full request body from save endpoint)
- `ON DELETE CASCADE` ensures states are deleted when evaluation runs are deleted
- `UNIQUE(evaluation_run_id, user_id)` prevents duplicate states
- Indexes optimize common queries (load by run_id, list by user_id)

---

## Background Cleanup Job

### Auto-Delete Expired States

**Job Name:** `cleanup_expired_evaluation_states`

**Schedule:** Run daily at 2:00 AM (low traffic time)

**Logic:**
```sql
DELETE FROM evaluation_states
WHERE last_updated < NOW() - INTERVAL '7 days';
```

**Implementation Options:**
1. **Cron job** (if using cron scheduler)
2. **Celery task** (if using Celery)
3. **Database trigger** (PostgreSQL scheduled job)
4. **API endpoint** called by external scheduler

**Logging:**
- Log number of records deleted
- Log any errors
- Alert if > 1000 records deleted (potential issue)

---

## Security Considerations

### 1. Data Isolation
- ✅ All queries filter by `user_id` from JWT token
- ✅ Users can only access their own states
- ✅ No cross-user data leakage possible

### 2. No Sensitive Data Stored
- ✅ No user auth tokens stored (security risk avoided)
- ✅ No passwords or credentials
- ✅ Only evaluation configuration and progress data

### 3. Storage Limits
- ⚠️ Monitor JSONB column size (could be large for 1000+ question evaluations)
- ⚠️ Consider pagination for list endpoint if users have many states
- ⚠️ Consider size limit validation (e.g., max 10MB per state)

### 4. Rate Limiting (Optional)
- Consider rate limiting save endpoint (max 1 save per 5 seconds per user)
- Prevents abuse from misbehaving clients

---

## Migration Plan

### Phase 1: Backend Implementation
1. Create database table
2. Implement 4 API endpoints
3. Add background cleanup job
4. Write unit tests
5. Deploy to staging

### Phase 2: Frontend Integration
1. Update Plugin to call new endpoints
2. Implement dual persistence (localStorage + backend)
3. Update resume logic (backend first, localStorage fallback)
4. Test resume flow across devices
5. Deploy to production

### Phase 3: Monitoring & Optimization
1. Monitor state storage size
2. Track resume success rate
3. Optimize indexes if needed
4. Add alerting for failures

---

## Testing Requirements

### Unit Tests
- ✅ Save state (new record)
- ✅ Save state (update existing)
- ✅ Load state (exists, not expired)
- ✅ Load state (exists, but expired → 404)
- ✅ Load state (doesn't exist → 404)
- ✅ Delete state (exists)
- ✅ Delete state (doesn't exist, still 204)
- ✅ List in-progress (multiple states)
- ✅ List in-progress (no states → empty array)
- ✅ User isolation (user A can't access user B's state)

### Integration Tests
- ✅ Save → Load → Verify data integrity
- ✅ Save → Wait 8 days (mock time) → Load → 404
- ✅ Save → Delete → Load → 404
- ✅ Cleanup job deletes only expired states

### Manual Testing
- ✅ Start evaluation → close browser → reopen → resume
- ✅ Start on laptop → resume on desktop (same user)
- ✅ Verify timestamps are correct
- ✅ Verify progress metadata is accurate

---

## Example Flow: Plugin + Backend Integration

### Start Evaluation
```typescript
// Plugin saves to both localStorage AND backend
EvaluationPersistence.saveState(state);  // localStorage
await evaluationApiService.saveEvaluationState(runId, state);  // backend
```

### After Each Batch
```typescript
// Update progress
this.processedQuestionIds.add(testCase.test_case_id);

// Save to both
this.savePersistenceState(runId, testCases);  // calls localStorage + backend
```

### Resume Evaluation
```typescript
// Try backend first
let state = await evaluationApiService.loadEvaluationState(runId);

if (!state) {
  // Fallback to localStorage
  state = EvaluationPersistence.loadState();
}

if (!state) {
  throw new Error('No evaluation state found');
}

// Continue evaluation...
```

### Completion/Error
```typescript
// Clear both
EvaluationPersistence.clearState();  // localStorage
await evaluationApiService.deleteEvaluationState(runId);  // backend
```

---

## Questions for Backend Team

1. **Database:** Confirm PostgreSQL version supports JSONB?
2. **Cleanup Job:** Preferred scheduling method (Celery, cron, other)?
3. **Storage Limits:** Should we enforce max JSONB size per state?
4. **Monitoring:** Do you have existing logging/alerting infrastructure?
5. **Timeline:** When can these endpoints be deployed to staging?

---

## Summary

**What:** 4 new REST API endpoints for evaluation state persistence

**Why:** Enable reliable resume of interrupted evaluations across sessions/devices

**How:** Backend stores evaluation progress as JSONB with 7-day retention

**Impact:**
- ✅ Better user experience (don't lose progress)
- ✅ Cross-device support (start on laptop, resume on desktop)
- ✅ More reliable than browser localStorage
- ✅ No security risks (no tokens stored)

**Next Steps:**
1. Backend team reviews this document
2. Backend team implements endpoints + database schema
3. Frontend team integrates new APIs
4. Test resume flow end-to-end
5. Deploy to production

---

**Document Version:** 1.0
**Created:** 2025-01-15
**Author:** BrainDrive Plugin Team
**Status:** Awaiting Backend Implementation
