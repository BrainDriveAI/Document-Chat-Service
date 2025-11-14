# API Reference for Frontend Integration

**Target Audience:** Frontend AI Coding Agents (Cursor, Windsurf, Cline, etc.)

This document provides complete API specifications for integrating with the Chat with Documents backend. All endpoints return JSON unless otherwise specified.

**Base URL:** `http://localhost:8000/api` (development)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Collections API](#collections-api)
3. [Documents API](#documents-api)
4. [Chat API](#chat-api)
5. [Search API](#search-api)
6. [Evaluation API](#evaluation-api)
7. [Health Check](#health-check)
8. [Error Handling](#error-handling)
9. [Common Schemas](#common-schemas)

---

## Authentication

**Current:** No authentication required (local deployment)

**Production:** Add `Authorization: Bearer <token>` header (future)

---

## Collections API

Collections group related documents for organized searching and chatting.

###

 **Base Path:** `/api/collections`

### Create Collection

Create a new document collection.

**Endpoint:** `POST /api/collections/`

**Request Body:**
```json
{
  "name": "Product Documentation",
  "description": "All product-related documentation",
  "color": "#3B82F6"
}
```

**Request Schema:**
```typescript
{
  name: string;          // Required, min length 1
  description: string;   // Required, min length 1
  color?: string;        // Optional, hex color (e.g., "#3B82F6")
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Product Documentation",
  "description": "All product-related documentation",
  "color": "#3B82F6",
  "created_at": "2025-11-14T12:00:00.000Z",
  "updated_at": "2025-11-14T12:00:00.000Z",
  "document_count": 0
}
```

**Example (fetch):**
```javascript
const response = await fetch('http://localhost:8000/api/collections/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Product Documentation',
    description: 'All product-related documentation',
    color: '#3B82F6'
  })
});
const collection = await response.json();
```

---

### List Collections

Get all collections.

**Endpoint:** `GET /api/collections/`

**Query Parameters:** None

**Response:** `200 OK`
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Product Documentation",
    "description": "All product-related documentation",
    "color": "#3B82F6",
    "created_at": "2025-11-14T12:00:00.000Z",
    "updated_at": "2025-11-14T12:00:00.000Z",
    "document_count": 5
  }
]
```

**Example (fetch):**
```javascript
const response = await fetch('http://localhost:8000/api/collections/');
const collections = await response.json();
```

---

### Get Collection

Get a single collection by ID.

**Endpoint:** `GET /api/collections/{collection_id}`

**Path Parameters:**
- `collection_id` (string, UUID): Collection ID

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Product Documentation",
  "description": "All product-related documentation",
  "color": "#3B82F6",
  "created_at": "2025-11-14T12:00:00.000Z",
  "updated_at": "2025-11-14T12:00:00.000Z",
  "document_count": 5
}
```

**Errors:**
- `404 Not Found`: Collection not found

**Example (fetch):**
```javascript
const response = await fetch(`http://localhost:8000/api/collections/${collectionId}`);
if (!response.ok) {
  if (response.status === 404) {
    console.error('Collection not found');
  }
  throw new Error('Failed to fetch collection');
}
const collection = await response.json();
```

---

### Update Collection

Update collection name, description, or color.

**Endpoint:** `PUT /api/collections/{collection_id}`

**Path Parameters:**
- `collection_id` (string, UUID): Collection ID

**Request Body:** (all fields optional)
```json
{
  "name": "Updated Name",
  "description": "Updated description",
  "color": "#10B981"
}
```

**Response:** `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Updated Name",
  "description": "Updated description",
  "color": "#10B981",
  "created_at": "2025-11-14T12:00:00.000Z",
  "updated_at": "2025-11-14T12:30:00.000Z",
  "document_count": 5
}
```

**Errors:**
- `404 Not Found`: Collection not found

**Example (fetch):**
```javascript
const response = await fetch(`http://localhost:8000/api/collections/${collectionId}`, {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Updated Name',
    description: 'Updated description'
  })
});
const updated = await response.json();
```

---

### Delete Collection

Delete a collection.

**Endpoint:** `DELETE /api/collections/{collection_id}`

**Path Parameters:**
- `collection_id` (string, UUID): Collection ID

**Response:** `204 No Content`

**Errors:**
- `404 Not Found`: Collection not found

**Example (fetch):**
```javascript
const response = await fetch(`http://localhost:8000/api/collections/${collectionId}`, {
  method: 'DELETE'
});
if (response.status === 204) {
  console.log('Collection deleted successfully');
}
```

---

## Documents API

Upload and manage documents within collections.

### **Base Path:** `/api/documents`

### Upload Document

Upload a document to a collection for processing.

**Endpoint:** `POST /api/documents/`

**Request:** `multipart/form-data`

**Form Fields:**
- `file` (File): Document file (PDF, DOCX, DOC, MD, HTML, PPTX)
- `collection_id` (string, UUID): Target collection ID

**Response:** `201 Created` (processing starts in background)
```json
{
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "filename": "doc_650e8400.pdf",
  "original_filename": "user-manual.pdf",
  "file_path": "/data/uploads/550e8400.../doc_650e8400.pdf",
  "file_size": 1024000,
  "document_type": "pdf",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "created_at": "2025-11-14T12:00:00.000Z",
  "processed_at": null,
  "metadata": {},
  "chunk_count": 0
}
```

**Document Status Values:**
- `pending`: Upload complete, awaiting processing
- `processing`: Currently being processed
- `completed`: Successfully processed
- `failed`: Processing failed

**Supported File Types:**
- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- PowerPoint (`.pptx`, `.ppt`)
- Markdown (`.md`, `.markdown`)
- HTML (`.html`, `.htm`)

**Example (fetch with FormData):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('collection_id', collectionId);

const response = await fetch('http://localhost:8000/api/documents/', {
  method: 'POST',
  body: formData  // Don't set Content-Type, browser sets it with boundary
});
const document = await response.json();

// Poll for processing completion
const checkStatus = async (docId) => {
  const res = await fetch(`http://localhost:8000/api/documents/${docId}`);
  const doc = await res.json();
  if (doc.status === 'completed') {
    console.log(`Document processed: ${doc.chunk_count} chunks`);
  } else if (doc.status === 'failed') {
    console.error('Document processing failed');
  } else {
    setTimeout(() => checkStatus(docId), 2000);  // Poll every 2s
  }
};
checkStatus(document.id);
```

---

### Get Document

Get document details and processing status.

**Endpoint:** `GET /api/documents/{document_id}`

**Path Parameters:**
- `document_id` (string, UUID): Document ID

**Response:** `200 OK`
```json
{
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "filename": "doc_650e8400.pdf",
  "original_filename": "user-manual.pdf",
  "file_path": "/data/uploads/550e8400.../doc_650e8400.pdf",
  "file_size": 1024000,
  "document_type": "pdf",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2025-11-14T12:00:00.000Z",
  "processed_at": "2025-11-14T12:02:30.000Z",
  "metadata": {
    "pages": 42,
    "processing_time_seconds": 150
  },
  "chunk_count": 87
}
```

**Errors:**
- `404 Not Found`: Document not found

---

### List Documents

List all documents, optionally filtered by collection.

**Endpoint:** `GET /api/documents/`

**Query Parameters:**
- `collection_id` (string, UUID, optional): Filter by collection

**Response:** `200 OK`
```json
[
  {
    "id": "650e8400-e29b-41d4-a716-446655440001",
    "filename": "doc_650e8400.pdf",
    "original_filename": "user-manual.pdf",
    "file_path": "/data/uploads/550e8400.../doc_650e8400.pdf",
    "file_size": 1024000,
    "document_type": "pdf",
    "collection_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "created_at": "2025-11-14T12:00:00.000Z",
    "processed_at": "2025-11-14T12:02:30.000Z",
    "metadata": {},
    "chunk_count": 87
  }
]
```

**Example (fetch with query param):**
```javascript
const response = await fetch(`http://localhost:8000/api/documents/?collection_id=${collectionId}`);
const documents = await response.json();
```

---

### Delete Document

Delete a document and its chunks from vector store.

**Endpoint:** `DELETE /api/documents/{document_id}`

**Path Parameters:**
- `document_id` (string, UUID): Document ID

**Response:** `204 No Content`

**Errors:**
- `404 Not Found`: Document not found
- `500 Internal Server Error`: Partial deletion (some chunks may remain)

**Example (fetch):**
```javascript
const response = await fetch(`http://localhost:8000/api/documents/${documentId}`, {
  method: 'DELETE'
});
if (response.status === 204) {
  console.log('Document deleted successfully');
}
```

---

## Chat API

Manage chat sessions and send messages with context retrieval.

### **Base Path:** `/api/chat`

### Create Chat Session

Create a new chat session, optionally linked to a collection.

**Endpoint:** `POST /api/chat/sessions`

**Request Body:**
```json
{
  "name": "Product Support Chat",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Request Schema:**
```typescript
{
  name: string;          // Required
  collection_id?: string; // Optional UUID
}
```

**Response:** `201 Created`
```json
{
  "id": "750e8400-e29b-41d4-a716-446655440002",
  "name": "Product Support Chat",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-11-14T12:00:00.000Z",
  "updated_at": "2025-11-14T12:00:00.000Z",
  "message_count": 0
}
```

**Example (fetch):**
```javascript
const response = await fetch('http://localhost:8000/api/chat/sessions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Product Support Chat',
    collection_id: collectionId
  })
});
const session = await response.json();
```

---

### List Chat Sessions

Get all chat sessions.

**Endpoint:** `GET /api/chat/sessions`

**Response:** `200 OK`
```json
[
  {
    "id": "750e8400-e29b-41d4-a716-446655440002",
    "name": "Product Support Chat",
    "collection_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-11-14T12:00:00.000Z",
    "updated_at": "2025-11-14T12:15:00.000Z",
    "message_count": 12
  }
]
```

---

### Delete Chat Session

Delete a chat session and all its messages.

**Endpoint:** `DELETE /api/chat/sessions/{session_id}`

**Path Parameters:**
- `session_id` (string, UUID): Session ID

**Response:** `204 No Content`

**Errors:**
- `404 Not Found`: Session not found

---

### Send Chat Message

Send a message and get AI response with context from documents.

**Endpoint:** `POST /api/chat/message`

**Request Body:**
```json
{
  "session_id": "750e8400-e29b-41d4-a716-446655440002",
  "message": "How do I reset my password?",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "top_k": 5,
  "use_hybrid_search": true
}
```

**Request Schema:**
```typescript
{
  session_id: string;           // Required UUID
  message: string;              // Required, user's question
  collection_id?: string;       // Optional UUID (overrides session default)
  top_k?: number;              // Optional, default 5, max chunks to retrieve
  use_hybrid_search?: boolean; // Optional, default true (vector + BM25)
}
```

**Response:** `200 OK`
```json
{
  "id": "850e8400-e29b-41d4-a716-446655440003",
  "session_id": "750e8400-e29b-41d4-a716-446655440002",
  "role": "assistant",
  "content": "To reset your password, go to Settings > Account > Security and click 'Reset Password'. You'll receive a confirmation email.",
  "metadata": {
    "intent": "retrieval",
    "chunks_retrieved": 3,
    "source_documents": [
      {
        "document_id": "650e8400-e29b-41d4-a716-446655440001",
        "chunk_id": "chunk_001",
        "content": "Password reset: Navigate to Settings...",
        "score": 0.92
      }
    ],
    "response_time_ms": 1523
  },
  "created_at": "2025-11-14T12:05:00.000Z",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example (fetch):**
```javascript
const response = await fetch('http://localhost:8000/api/chat/message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: sessionId,
    message: 'How do I reset my password?',
    collection_id: collectionId,
    top_k: 5
  })
});
const reply = await response.json();
console.log('AI:', reply.content);
console.log('Sources:', reply.metadata.source_documents);
```

---

### Get Chat History

Get message history for a session.

**Endpoint:** `GET /api/chat/messages`

**Query Parameters:**
- `session_id` (string, UUID, required): Session ID
- `limit` (number, optional): Max messages to return (default: 50)

**Response:** `200 OK`
```json
[
  {
    "id": "850e8400-e29b-41d4-a716-446655440003",
    "session_id": "750e8400-e29b-41d4-a716-446655440002",
    "role": "user",
    "content": "How do I reset my password?",
    "metadata": {},
    "created_at": "2025-11-14T12:04:55.000Z",
    "collection_id": "550e8400-e29b-41d4-a716-446655440000"
  },
  {
    "id": "850e8400-e29b-41d4-a716-446655440004",
    "session_id": "750e8400-e29b-41d4-a716-446655440002",
    "role": "assistant",
    "content": "To reset your password...",
    "metadata": {
      "chunks_retrieved": 3,
      "response_time_ms": 1523
    },
    "created_at": "2025-11-14T12:05:00.000Z",
    "collection_id": "550e8400-e29b-41d4-a716-446655440000"
  }
]
```

**Example (fetch):**
```javascript
const response = await fetch(`http://localhost:8000/api/chat/messages?session_id=${sessionId}&limit=20`);
const messages = await response.json();
```

---

## Search API

Direct search without chat context (retrieve document chunks).

### **Base Path:** `/api/search_u`

### Search Documents

Search for relevant document chunks using hybrid search.

**Endpoint:** `POST /api/search_u/`

**Request Body:**
```json
{
  "query": "password reset procedure",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "top_k": 5,
  "use_hybrid_search": true,
  "similarity_threshold": 0.7
}
```

**Request Schema:**
```typescript
{
  query: string;                // Required, search query
  collection_id?: string;       // Optional UUID
  top_k?: number;              // Optional, default 5
  use_hybrid_search?: boolean; // Optional, default true
  similarity_threshold?: number; // Optional, 0-1, default 0.7
}
```

**Response:** `200 OK`
```json
{
  "chunks": [
    {
      "id": "chunk_001",
      "document_id": "650e8400-e29b-41d4-a716-446655440001",
      "content": "To reset your password, navigate to Settings > Account > Security...",
      "metadata": {
        "page": 15,
        "heading": "Security Settings"
      },
      "score": 0.92
    }
  ],
  "intent": {
    "kind": "retrieval",
    "confidence": 0.95,
    "reasoning": "User query seeks specific information"
  },
  "requires_generation": false,
  "generation_type": null,
  "metadata": {
    "queries_used": ["password reset procedure"],
    "search_method": "hybrid",
    "total_results": 5
  }
}
```

**Example (fetch):**
```javascript
const response = await fetch('http://localhost:8000/api/search_u/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'password reset procedure',
    collection_id: collectionId,
    top_k: 5
  })
});
const results = await response.json();
console.log('Found chunks:', results.chunks.length);
```

---

## Evaluation API

Run and manage RAG evaluation tests.

### **Base Path:** `/api/evaluation`

### Start Evaluation Run

Start a new evaluation run with test cases.

**Endpoint:** `POST /api/evaluation/run`

**Request Body:**
```json
{
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "test_cases_path": "evaluation_test_docs/test_cases.json",
  "config": {
    "top_k": 5,
    "use_hybrid_search": true
  }
}
```

**Response:** `202 Accepted` (background task started)
```json
{
  "evaluation_id": "950e8400-e29b-41d4-a716-446655440005",
  "status": "running",
  "message": "Evaluation started"
}
```

---

### Get Evaluation Results

Get evaluation run status and results.

**Endpoint:** `GET /api/evaluation/results/{evaluation_id}`

**Path Parameters:**
- `evaluation_id` (string, UUID): Evaluation run ID

**Response:** `200 OK`
```json
{
  "id": "950e8400-e29b-41d4-a716-446655440005",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_questions": 20,
  "evaluated_count": 20,
  "correct_count": 18,
  "incorrect_count": 2,
  "accuracy": 0.90,
  "is_completed": true,
  "config": {
    "top_k": 5,
    "use_hybrid_search": true
  },
  "created_at": "2025-11-14T12:00:00.000Z",
  "completed_at": "2025-11-14T12:15:00.000Z",
  "results": [
    {
      "question": "How do I reset my password?",
      "answer": "Navigate to Settings > Account...",
      "is_correct": true,
      "confidence": 0.95,
      "chunks_used": 3
    }
  ]
}
```

**Status Values:**
- `pending`: Queued
- `running`: In progress
- `completed`: Finished successfully
- `failed`: Error occurred

---

### Plugin Evaluation - Start

Start evaluation for plugin integration (frontend-driven flow).

**Endpoint:** `POST /api/evaluation/plugin/start`

**Request Body:**
```json
{
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_questions": 20,
  "user_id": "user_123",
  "config": {
    "top_k": 5
  }
}
```

**Response:** `200 OK`
```json
{
  "evaluation_run_id": "a50e8400-e29b-41d4-a716-446655440006",
  "collection_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_questions": 20,
  "status": "in_progress"
}
```

---

### Plugin Evaluation - Submit Results

Submit evaluation results from frontend (batch).

**Endpoint:** `POST /api/evaluation/plugin/submit`

**Request Body:**
```json
{
  "evaluation_run_id": "a50e8400-e29b-41d4-a716-446655440006",
  "results": [
    {
      "question": "How to reset password?",
      "generated_answer": "Go to Settings...",
      "retrieved_chunks": ["chunk1", "chunk2"],
      "is_correct": true,
      "evaluation_metadata": {
        "judge_reasoning": "Answer is factual"
      }
    }
  ]
}
```

**Response:** `200 OK`
```json
{
  "evaluation_run_id": "a50e8400-e29b-41d4-a716-446655440006",
  "results_saved": 1,
  "new_evaluated_count": 15,
  "is_completed": false
}
```

---

### Save Evaluation State

Save evaluation progress (for resume later).

**Endpoint:** `POST /api/evaluation/state/{evaluation_run_id}`

**Path Parameters:**
- `evaluation_run_id` (string, UUID): Evaluation run ID

**Request Body:**
```json
{
  "current_question_index": 5,
  "answered_questions": 4,
  "user_answers": {
    "0": "user answer 1",
    "1": "user answer 2"
  },
  "evaluation_results": [...]
}
```

**Response:** `200 OK`
```json
{
  "evaluation_run_id": "a50e8400-e29b-41d4-a716-446655440006",
  "saved": true
}
```

---

### Load Evaluation State

Resume evaluation from saved state.

**Endpoint:** `GET /api/evaluation/state/{evaluation_run_id}`

**Response:** `200 OK`
```json
{
  "evaluation_run_id": "a50e8400-e29b-41d4-a716-446655440006",
  "state": {
    "current_question_index": 5,
    "answered_questions": 4,
    "user_answers": {...},
    "evaluation_results": [...]
  },
  "found": true
}
```

---

## Health Check

### Check API Health

**Endpoint:** `GET /api/health`

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2025-11-14T12:00:00.000Z",
  "version": "1.0.0"
}
```

**Example (fetch):**
```javascript
const response = await fetch('http://localhost:8000/api/health');
const health = await response.json();
console.log('API status:', health.status);
```

---

## Error Handling

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| `200` | OK | Successful GET/PUT request |
| `201` | Created | Successful POST (resource created) |
| `202` | Accepted | Background task started |
| `204` | No Content | Successful DELETE |
| `400` | Bad Request | Invalid request data |
| `404` | Not Found | Resource doesn't exist |
| `500` | Internal Server Error | Server-side error |

### Error Handling Example

```javascript
async function fetchWithErrorHandling(url, options) {
  try {
    const response = await fetch(url, options);

    if (!response.ok) {
      const error = await response.json();

      switch (response.status) {
        case 400:
          throw new Error(`Invalid request: ${error.detail}`);
        case 404:
          throw new Error(`Not found: ${error.detail}`);
        case 500:
          throw new Error(`Server error: ${error.detail}`);
        default:
          throw new Error(`HTTP ${response.status}: ${error.detail}`);
      }
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}

// Usage
try {
  const collection = await fetchWithErrorHandling(
    'http://localhost:8000/api/collections/123',
    { method: 'GET' }
  );
} catch (error) {
  // Handle error in UI
  showErrorToast(error.message);
}
```

---

## Common Schemas

### UUID Format

All IDs are UUIDv4 strings:
```typescript
type UUID = string; // e.g., "550e8400-e29b-41d4-a716-446655440000"
```

### Datetime Format

All timestamps are ISO 8601 strings:
```typescript
type DateTime = string; // e.g., "2025-11-14T12:00:00.000Z"
```

### Color Format

Hex color codes:
```typescript
type HexColor = string; // e.g., "#3B82F6" or "#10B981"
```

### Document Status Enum

```typescript
type DocumentStatus =
  | "pending"      // Uploaded, awaiting processing
  | "processing"   // Currently being processed
  | "completed"    // Successfully processed
  | "failed";      // Processing failed
```

### Chat Role Enum

```typescript
type ChatRole =
  | "user"         // User message
  | "assistant"    // AI response
  | "system";      // System message
```

### Search Intent Enum

```typescript
type IntentKind =
  | "chat"                // Casual chat, no retrieval
  | "retrieval"           // Document search
  | "collection_summary"  // Summarize collection
  | "comparison"          // Compare concepts
  | "listing"             // Generate list
  | "clarification";      // Follow-up question
```

---

## Quick Integration Checklist

**For Frontend AI Agents:**

1. **Collections:**
   - [ ] Create collection: `POST /api/collections/`
   - [ ] List collections: `GET /api/collections/`
   - [ ] Get collection: `GET /api/collections/{id}`
   - [ ] Update collection: `PUT /api/collections/{id}`
   - [ ] Delete collection: `DELETE /api/collections/{id}`

2. **Documents:**
   - [ ] Upload document: `POST /api/documents/` (multipart/form-data)
   - [ ] Get document status: `GET /api/documents/{id}`
   - [ ] List documents: `GET /api/documents/?collection_id={id}`
   - [ ] Delete document: `DELETE /api/documents/{id}`
   - [ ] Poll for `status === "completed"` after upload

3. **Chat:**
   - [ ] Create session: `POST /api/chat/sessions`
   - [ ] Send message: `POST /api/chat/message`
   - [ ] Get history: `GET /api/chat/messages?session_id={id}`
   - [ ] Delete session: `DELETE /api/chat/sessions/{id}`
   - [ ] Display `metadata.source_documents` for citations

4. **Error Handling:**
   - [ ] Handle 404 (not found)
   - [ ] Handle 400 (bad request)
   - [ ] Handle 500 (server error)
   - [ ] Parse `detail` field in error responses

5. **Polling:**
   - [ ] Poll document status every 2-5s until `completed`
   - [ ] Poll evaluation results until `is_completed: true`

---

## Complete Integration Example

```javascript
// Complete workflow: Create collection, upload document, chat

async function completeWorkflow() {
  // 1. Create collection
  const collection = await fetch('http://localhost:8000/api/collections/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'My Documents',
      description: 'Personal documentation',
      color: '#3B82F6'
    })
  }).then(r => r.json());

  console.log('Collection created:', collection.id);

  // 2. Upload document
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('collection_id', collection.id);

  const document = await fetch('http://localhost:8000/api/documents/', {
    method: 'POST',
    body: formData
  }).then(r => r.json());

  console.log('Document uploaded:', document.id);

  // 3. Poll for processing completion
  let processed = false;
  while (!processed) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    const doc = await fetch(`http://localhost:8000/api/documents/${document.id}`)
      .then(r => r.json());

    if (doc.status === 'completed') {
      console.log(`Processing complete: ${doc.chunk_count} chunks`);
      processed = true;
    } else if (doc.status === 'failed') {
      throw new Error('Document processing failed');
    }
  }

  // 4. Create chat session
  const session = await fetch('http://localhost:8000/api/chat/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'Support Chat',
      collection_id: collection.id
    })
  }).then(r => r.json());

  console.log('Chat session created:', session.id);

  // 5. Send chat message
  const reply = await fetch('http://localhost:8000/api/chat/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: session.id,
      message: 'What is this document about?',
      collection_id: collection.id
    })
  }).then(r => r.json());

  console.log('AI Reply:', reply.content);
  console.log('Sources:', reply.metadata.source_documents);
}
```

---

## OpenAPI / Swagger Documentation

Interactive API documentation available at:
**`http://localhost:8000/docs`**

Features:
- Try out endpoints directly
- View request/response schemas
- Download OpenAPI spec

---

## Support

**Issues:** https://github.com/BrainDriveAI/Document-Chat-Service/issues
**Documentation:** `docs/OWNERS-MANUAL.md`
**Architecture:** `docs/braindrive_rag_system.md`

---

**Last Updated:** 2025-11-14
**API Version:** 1.0.0
