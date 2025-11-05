# Plugin Evaluation Integration Guide

## Overview

This guide explains how to integrate the RAG system evaluation workflow into a React plugin. The plugin-based evaluation workflow enables external applications to:

1. **Start an evaluation** - Retrieve test questions with pre-fetched context
2. **Generate answers** - Use the host system's LLM to generate answers
3. **Submit for judging** - Send answers back to backend for evaluation
4. **Track progress** - Monitor evaluation progress in real-time

## Architecture

The plugin evaluation workflow uses a **two-step process**:

```
┌─────────────────────────────────────────────────────────────┐
│  Plugin (React Application)                                 │
│                                                              │
│  1. Start Evaluation → Get Questions + Context              │
│  2. Generate Answers (using host LLM)                       │
│  3. Submit Answers → Backend judges and stores results      │
│  4. Repeat step 3 in batches for parallel processing        │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│  Backend API                                                 │
│                                                              │
│  • POST /api/evaluation/plugin/start                        │
│  • POST /api/evaluation/plugin/submit                       │
└─────────────────────────────────────────────────────────────┘
```

### Why Two Steps?

- **LLM Generation in Host**: Plugins can use the host system's LLM (not backend)
- **Incremental Submissions**: Submit answers in batches (e.g., every 3 answers)
- **Parallel Processing**: Backend judges answers while plugin generates more
- **Idempotent**: Duplicate submissions are automatically skipped

## API Reference

### 1. Start Plugin Evaluation

**Endpoint:** `POST /api/evaluation/plugin/start`

**Description:** Initializes a new evaluation run, retrieves context for all test questions, and returns test data for the plugin to process.

**Request:**
```http
POST http://localhost:8000/api/evaluation/plugin/start
Content-Type: application/json
```

**Response:**
```json
{
  "evaluation_run_id": "123e4567-e89b-12d3-a456-426614174000",
  "test_data": [
    {
      "test_case_id": "tc_001",
      "question": "What is BrainDrive?",
      "category": "general",
      "retrieved_context": "BrainDrive is an AI-powered document management platform that enables users to interact with their documents using natural language. It supports document upload, semantic search, and conversational AI.",
      "ground_truth": "BrainDrive is a platform that enables users to interact with documents using AI."
    },
    {
      "test_case_id": "tc_002",
      "question": "How does BrainDrive handle document processing?",
      "category": "technical",
      "retrieved_context": "BrainDrive uses a remote Document Processor API with spaCy Layout-based processing. Documents are chunked using hierarchical strategies and embedded using Ollama.",
      "ground_truth": null
    }
  ]
}
```

**Response Fields:**
- `evaluation_run_id` (string): Unique ID for this evaluation run. Use this for all subsequent submissions.
- `test_data` (array): Array of test questions with pre-fetched context
  - `test_case_id` (string): Unique identifier for this test case
  - `question` (string): Question to answer
  - `category` (string): Question category (e.g., "general", "technical")
  - `retrieved_context` (string): Pre-fetched relevant context from vector store
  - `ground_truth` (string|null): Optional expected answer for reference

**Error Responses:**

```json
// 404 - Test cases file not found
{
  "detail": "Test cases file not found: /path/to/test_cases.json"
}

// 400 - Test collection not initialized
{
  "detail": "Evaluation test collection not found: eval-test-collection-..."
}

// 500 - Server error
{
  "detail": "Failed to start plugin evaluation: ..."
}
```

---

### 2. Submit Plugin Evaluation

**Endpoint:** `POST /api/evaluation/plugin/submit`

**Description:** Submits LLM-generated answers for evaluation. Backend judges each answer using OpenAI GPT and stores results. Supports incremental batch submissions (idempotent).

**Request:**
```http
POST http://localhost:8000/api/evaluation/plugin/submit
Content-Type: application/json

{
  "evaluation_run_id": "123e4567-e89b-12d3-a456-426614174000",
  "submissions": [
    {
      "test_case_id": "tc_001",
      "llm_answer": "BrainDrive is an AI-powered platform that allows users to interact with their documents using natural language processing and semantic search capabilities.",
      "retrieved_context": "BrainDrive is an AI-powered document management platform that enables users to interact with their documents using natural language. It supports document upload, semantic search, and conversational AI."
    },
    {
      "test_case_id": "tc_002",
      "llm_answer": "BrainDrive processes documents by sending them to a remote Document Processor API which uses spaCy for layout analysis. The documents are then chunked hierarchically and converted to embeddings using Ollama models.",
      "retrieved_context": "BrainDrive uses a remote Document Processor API with spaCy Layout-based processing. Documents are chunked using hierarchical strategies and embedded using Ollama."
    }
  ]
}
```

**Request Fields:**
- `evaluation_run_id` (string, required): ID from start endpoint
- `submissions` (array, required): Array of answer submissions
  - `test_case_id` (string, required): ID of the test case being answered
  - `llm_answer` (string, required): LLM-generated answer to the question
  - `retrieved_context` (string, required): Context used (from start endpoint)

**Response:**
```json
{
  "evaluation_run_id": "123e4567-e89b-12d3-a456-426614174000",
  "processed_count": 2,
  "skipped_count": 0,
  "total_evaluated": 2,
  "total_questions": 30,
  "progress": 0.067,
  "is_completed": false,
  "correct_count": 1,
  "incorrect_count": 1
}
```

**Response Fields:**
- `evaluation_run_id` (string): Echo of the evaluation run ID
- `processed_count` (number): Number of new submissions processed in this request
- `skipped_count` (number): Number of submissions skipped (already evaluated)
- `total_evaluated` (number): Total questions evaluated so far
- `total_questions` (number): Total questions in evaluation
- `progress` (number): Evaluation progress (0.0 to 1.0)
- `is_completed` (boolean): Whether all questions have been evaluated
- `correct_count` (number): Total correct answers so far
- `incorrect_count` (number): Total incorrect answers so far

**Idempotent Behavior:**

If you submit the same `test_case_id` twice:

```json
// First submission
{
  "processed_count": 1,
  "skipped_count": 0,
  "total_evaluated": 1
}

// Second submission (same test_case_id)
{
  "processed_count": 0,
  "skipped_count": 1,
  "total_evaluated": 1  // Still 1
}
```

**Error Responses:**

```json
// 400 - Evaluation run not found
{
  "detail": "Evaluation run not found: 123e4567-..."
}

// 404 - Test cases file not found
{
  "detail": "Test cases file not found: /path/to/test_cases.json"
}

// 500 - Server error
{
  "detail": "Failed to submit plugin evaluation: ..."
}
```

---

## Integration Guide

### Step 1: Setup TypeScript Types

Create type definitions for the API:

```typescript
// types/evaluation.ts

export interface TestDataItem {
  test_case_id: string;
  question: string;
  category: string;
  retrieved_context: string;
  ground_truth?: string;
}

export interface StartEvaluationResponse {
  evaluation_run_id: string;
  test_data: TestDataItem[];
}

export interface SubmissionItem {
  test_case_id: string;
  llm_answer: string;
  retrieved_context: string;
}

export interface SubmitEvaluationRequest {
  evaluation_run_id: string;
  submissions: SubmissionItem[];
}

export interface SubmitEvaluationResponse {
  evaluation_run_id: string;
  processed_count: number;
  skipped_count: number;
  total_evaluated: number;
  total_questions: number;
  progress: number;
  is_completed: boolean;
  correct_count: number;
  incorrect_count: number;
}
```

### Step 2: Create API Client

```typescript
// services/evaluationApi.ts

const API_BASE_URL = 'http://localhost:8000/api/evaluation';

export async function startPluginEvaluation(): Promise<StartEvaluationResponse> {
  const response = await fetch(`${API_BASE_URL}/plugin/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start evaluation');
  }

  return response.json();
}

export async function submitPluginEvaluation(
  request: SubmitEvaluationRequest
): Promise<SubmitEvaluationResponse> {
  const response = await fetch(`${API_BASE_URL}/plugin/submit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to submit evaluation');
  }

  return response.json();
}
```

### Step 3: Implement Evaluation Logic

```typescript
// hooks/useEvaluation.ts

import { useState } from 'react';
import { startPluginEvaluation, submitPluginEvaluation } from '../services/evaluationApi';
import type { TestDataItem, SubmissionItem } from '../types/evaluation';

// Replace with your actual LLM service
async function generateLLMAnswer(question: string, context: string): Promise<string> {
  // Example: Call to host system's LLM
  // This could be OpenAI, Anthropic, or local model
  const response = await fetch('YOUR_LLM_ENDPOINT', {
    method: 'POST',
    body: JSON.stringify({
      prompt: `Context: ${context}\n\nQuestion: ${question}\n\nAnswer:`,
    }),
  });

  const data = await response.json();
  return data.answer;
}

export function useEvaluation() {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const runEvaluation = async () => {
    setIsRunning(true);
    setError(null);
    setProgress(0);

    try {
      // Step 1: Start evaluation and get test data
      console.log('Starting evaluation...');
      const { evaluation_run_id, test_data } = await startPluginEvaluation();
      console.log(`Evaluation started: ${evaluation_run_id}`);
      console.log(`Total questions: ${test_data.length}`);

      // Step 2: Process questions in batches
      const BATCH_SIZE = 3;
      const batches: TestDataItem[][] = [];

      for (let i = 0; i < test_data.length; i += BATCH_SIZE) {
        batches.push(test_data.slice(i, i + BATCH_SIZE));
      }

      let allCorrect = 0;
      let allIncorrect = 0;

      for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
        const batch = batches[batchIndex];
        console.log(`Processing batch ${batchIndex + 1}/${batches.length}`);

        // Step 3: Generate LLM answers for this batch
        const submissions: SubmissionItem[] = await Promise.all(
          batch.map(async (testCase) => {
            console.log(`Generating answer for: ${testCase.question}`);
            const llm_answer = await generateLLMAnswer(
              testCase.question,
              testCase.retrieved_context
            );

            return {
              test_case_id: testCase.test_case_id,
              llm_answer,
              retrieved_context: testCase.retrieved_context,
            };
          })
        );

        // Step 4: Submit batch for judging
        console.log(`Submitting ${submissions.length} answers...`);
        const result = await submitPluginEvaluation({
          evaluation_run_id,
          submissions,
        });

        console.log(`Batch ${batchIndex + 1} results:`, result);

        // Update progress
        setProgress(result.progress);
        allCorrect = result.correct_count;
        allIncorrect = result.incorrect_count;

        // Check if completed
        if (result.is_completed) {
          console.log('Evaluation completed!');
          setResults({
            evaluation_run_id,
            total_questions: result.total_questions,
            correct_count: result.correct_count,
            incorrect_count: result.incorrect_count,
            accuracy: (result.correct_count / result.total_questions) * 100,
          });
          break;
        }
      }

      setIsRunning(false);
    } catch (err) {
      console.error('Evaluation failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsRunning(false);
    }
  };

  return {
    runEvaluation,
    isRunning,
    progress,
    results,
    error,
  };
}
```

### Step 4: Create UI Component

```tsx
// components/EvaluationPanel.tsx

import React from 'react';
import { useEvaluation } from '../hooks/useEvaluation';

export function EvaluationPanel() {
  const { runEvaluation, isRunning, progress, results, error } = useEvaluation();

  return (
    <div className="evaluation-panel">
      <h2>RAG System Evaluation</h2>

      {!isRunning && !results && (
        <button onClick={runEvaluation} className="btn-primary">
          Run Evaluation
        </button>
      )}

      {isRunning && (
        <div className="progress-container">
          <p>Evaluation in progress...</p>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
          <p>{Math.round(progress * 100)}% complete</p>
        </div>
      )}

      {results && (
        <div className="results-container">
          <h3>Evaluation Complete!</h3>
          <div className="stats">
            <div className="stat-card">
              <label>Accuracy</label>
              <value>{results.accuracy.toFixed(2)}%</value>
            </div>
            <div className="stat-card">
              <label>Correct</label>
              <value>{results.correct_count} / {results.total_questions}</value>
            </div>
            <div className="stat-card">
              <label>Incorrect</label>
              <value>{results.incorrect_count}</value>
            </div>
          </div>
          <p className="evaluation-id">
            Evaluation ID: {results.evaluation_run_id}
          </p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <p>Error: {error}</p>
        </div>
      )}
    </div>
  );
}
```

## Advanced Usage

### Custom Batch Size

Adjust batch size based on your LLM's speed:

```typescript
const BATCH_SIZE = 5; // Process 5 questions at a time

// Faster LLM? Use larger batches
// Slower LLM? Use smaller batches (1-3)
```

### Parallel LLM Generation

Generate multiple answers in parallel:

```typescript
// Generate all answers in batch concurrently
const submissions = await Promise.all(
  batch.map(async (testCase) => {
    const answer = await generateLLMAnswer(
      testCase.question,
      testCase.retrieved_context
    );
    return { test_case_id: testCase.test_case_id, llm_answer: answer, ... };
  })
);
```

### Resume Interrupted Evaluation

The workflow is idempotent, so you can resume:

```typescript
// If evaluation was interrupted, just resubmit all answers
// Already-evaluated answers will be skipped automatically
const result = await submitPluginEvaluation({
  evaluation_run_id: 'previous-run-id',
  submissions: allSubmissions, // Backend skips duplicates
});

console.log(`Skipped ${result.skipped_count} already-evaluated answers`);
```

### View Results in Web UI

After completion, view detailed results in the web interface:

```
http://localhost:8000/evaluation
```

Click on your evaluation run to see:
- Individual question results
- Judge reasoning for each answer
- Factual errors detected
- Missing information

## Best Practices

1. **Use Batch Submission**: Submit every 3-5 answers for parallel processing
2. **Handle Errors**: Wrap API calls in try-catch and show user-friendly errors
3. **Show Progress**: Update UI with real-time progress (0.0 to 1.0)
4. **Log Everything**: Console.log helps debug LLM generation issues
5. **Test Idempotency**: Verify duplicate submissions are handled correctly

## Configuration

Backend configuration (`.env`):

```bash
# Evaluation settings
INITIALIZE_TEST_COLLECTION=false
EVALUATION_TEST_COLLECTION_ID=eval-test-collection-00000000-0000-0000-0000-000000000001

# OpenAI judge service
OPENAI_EVALUATION_API_KEY=sk-...
OPENAI_EVALUATION_MODEL=gpt-4
OPENAI_EVALUATION_TIMEOUT=60

# Test data location
EVALUATION_TEST_DOCS_DIR=./evaluation_test_docs
```

## Troubleshooting

### "Test collection not found"
Ensure `INITIALIZE_TEST_COLLECTION=true` and restart backend to initialize test collection.

### "Evaluation run not found"
Check `evaluation_run_id` is correct. Each run has unique ID from start endpoint.

### Slow judging
OpenAI GPT judging can be slow. Consider:
- Using faster model (gpt-3.5-turbo instead of gpt-4)
- Increasing `OPENAI_EVALUATION_TIMEOUT`
- Reducing batch size

### Progress stuck at < 1.0
Check for errors in backend logs. Some submissions may have failed silently.

## Example: Complete Workflow

```typescript
// 1. Start evaluation
const { evaluation_run_id, test_data } = await startPluginEvaluation();
// => 30 test questions with context

// 2. Generate answers (batch 1: questions 1-3)
const batch1 = test_data.slice(0, 3);
const answers1 = await Promise.all(
  batch1.map(q => generateLLMAnswer(q.question, q.retrieved_context))
);

// 3. Submit batch 1
const result1 = await submitPluginEvaluation({
  evaluation_run_id,
  submissions: batch1.map((q, i) => ({
    test_case_id: q.test_case_id,
    llm_answer: answers1[i],
    retrieved_context: q.retrieved_context,
  })),
});
// => { processed_count: 3, progress: 0.1, ... }

// 4. Continue with remaining batches until is_completed: true
```

## Support

For issues or questions:
- Check backend logs: `docker-compose logs -f`
- View evaluation results: http://localhost:8000/evaluation
- API documentation: http://localhost:8000/docs
