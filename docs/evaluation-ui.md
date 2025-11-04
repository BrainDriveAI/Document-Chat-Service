# Evaluation UI Documentation

## Overview

The evaluation UI provides a comprehensive web interface for running and viewing RAG system evaluations. It's accessible at `/evaluation` and features a clean, modern design with dark mode support.

## Features

### 1. Two-Tab Interface

**Evaluation Runs Tab**
- View list of all evaluation runs
- Real-time stats cards showing latest metrics
- Quick access to detailed results
- Visual indicators for accuracy (color-coded progress bars)
- Click any run to view detailed results

**Results Details Tab**
- Detailed summary of selected evaluation run
- Individual test case results with expand/collapse
- Search and filter functionality
- Visual pass/fail indicators
- Judge reasoning and error analysis

### 2. Run New Evaluation

- One-click evaluation trigger via "Run New Evaluation" button
- Real-time progress indicator
- Automatic navigation to results after completion
- Error handling with user-friendly messages

### 3. Stats Dashboard

Four key metrics displayed as cards:
- **Latest Accuracy**: Overall accuracy percentage
- **Correct Answers**: Count of correct vs total questions
- **Incorrect Answers**: Count of failures
- **Duration**: Time taken to complete evaluation

### 4. Results Table Features

**Runs List:**
- Date/time of evaluation
- Status badge (Completed/Running/Failed/Pending)
- Accuracy with visual progress bar
- Correct/Total count
- Duration in seconds
- "View Details" action button

**Results Details:**
- Expandable test case cards
- Color-coded border (green = correct, red = incorrect)
- Question and LLM answer display
- Judge verdict with reasoning
- Factual errors list (if any)
- Missing information list (if any)
- Retrieved context (collapsible)
- Ground truth display (if available)

### 5. Filtering and Search

- **Search**: Filter questions by text content
- **Correctness Filter**: Show all/correct only/incorrect only
- Real-time filtering without page reload

## User Experience Highlights

### Visual Design
- Clean, modern interface using Tailwind CSS
- Dark mode support with theme toggle
- Color-coded indicators:
  - Green: Correct answers, success states
  - Red: Incorrect answers, errors
  - Yellow: Warnings, in-progress states
  - Blue: Actions, information
- Lucide icons for intuitive navigation

### Interaction Patterns
- **Loading States**: Spinner overlay during evaluation
- **Toast Notifications**: Success/error messages
- **Expandable Content**: Click to reveal detailed information
- **Hover Effects**: Visual feedback on interactive elements
- **Smooth Transitions**: Animated tab switching and card expansion

### Responsive Design
- Mobile-friendly layout
- Adaptive grid for stats cards
- Scrollable tables on small screens
- Touch-friendly tap targets

## Page Sections

### Header
```
┌─────────────────────────────────────────────────┐
│ RAG System Evaluation    [Run New Evaluation]  │
│ Test and measure the accuracy of your RAG sys  │
└─────────────────────────────────────────────────┘
```

### Tabs
```
┌─────────────────────────────────────────────────┐
│ [Evaluation Runs] | [Results Details]           │
└─────────────────────────────────────────────────┘
```

### Stats Cards (2x2 grid on desktop)
```
┌─────────────┬─────────────┬─────────────┬──────────────┐
│  Accuracy   │   Correct   │  Incorrect  │   Duration   │
│    85.2%    │   25 / 30   │      5      │    45.2s     │
└─────────────┴─────────────┴─────────────┴──────────────┘
```

### Runs Table
```
┌─────────────────────────────────────────────────────────┐
│ Date         Status    Accuracy  Correct/Total Duration │
├─────────────────────────────────────────────────────────┤
│ Jan 4, 2:30  ✓ Done   ███ 85%   25/30         45.2s    │
│ Jan 3, 10:15 ✓ Done   ██░ 75%   22/30         42.1s    │
└─────────────────────────────────────────────────────────┘
```

### Results Detail Card (Expandable)
```
┌─────────────────────────────────────────────────┐
│ ✓ What is BrainDrive?                      [v]  │  ← Click to expand
├─────────────────────────────────────────────────┤
│ Question: What is BrainDrive?                   │
│                                                  │
│ LLM Answer:                                     │
│ ┌─────────────────────────────────────────────┐ │
│ │ BrainDrive is an open-source platform...   │ │
│ └─────────────────────────────────────────────┘ │
│                                                  │
│ Judge Verdict: [Correct]                        │
│ The answer accurately describes BrainDrive...   │
│                                                  │
│ [View Retrieved Context]                        │
└─────────────────────────────────────────────────┘
```

## Navigation

The Evaluation page is accessible from the main navigation bar:

```
[Dashboard] [Chat] [Search] [Evaluation]
```

Click "Evaluation" to access the evaluation interface.

## API Endpoints Used

- `POST /api/evaluation/run` - Trigger new evaluation
- `GET /api/evaluation/runs` - List all evaluation runs
- `GET /api/evaluation/results/{id}` - Get detailed results

## Technical Implementation

### Frontend Technologies
- **HTML/Jinja2**: Template rendering
- **Tailwind CSS**: Styling and responsive design
- **Lucide Icons**: Icon library
- **Vanilla JavaScript**: Interactivity and API calls

### Key JavaScript Functions
- `runEvaluation()` - Trigger evaluation run
- `loadRuns()` - Fetch and display evaluation runs
- `loadResults()` - Fetch and display detailed results
- `filterResults()` - Search and filter test results
- `toggleResult()` - Expand/collapse result details
- `renderSummary()` - Render evaluation summary card
- `renderResults()` - Render test case results list

### State Management
- `currentEvaluationId` - Currently selected evaluation
- `allResults` - Cached results for filtering

## Usage Flow

1. **User lands on /evaluation**
   - System loads recent evaluation runs
   - Displays stats from latest run

2. **User clicks "Run New Evaluation"**
   - Loading indicator appears
   - POST request to `/api/evaluation/run`
   - Toast notification shows progress
   - On completion, auto-switches to Results tab

3. **User views results**
   - Summary card shows overall metrics
   - Test cases listed with pass/fail indicators
   - Click any test case to expand details

4. **User filters results**
   - Type in search box to filter by question text
   - Select correctness filter for correct/incorrect only
   - Results update in real-time

5. **User reviews past runs**
   - Switch to "Evaluation Runs" tab
   - Click any run row to view its results
   - Compare accuracy across different runs

## Customization

### Colors
Modify Tailwind classes to change color scheme:
- Success: `green-*` classes
- Error: `red-*` classes
- Info: `blue-*` classes
- Warning: `yellow-*` classes

### Layout
Adjust grid columns in stats cards:
```html
<div class="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
```

### Pagination
Add pagination to runs table for large datasets:
```javascript
// Add page parameter to API call
const data = await utils.apiRequest('/api/evaluation/runs?limit=50&offset=0');
```

## Future Enhancements

- Export results to CSV/JSON
- Comparison view (compare multiple runs side-by-side)
- Category-based filtering and grouping
- Accuracy trend chart over time
- Real-time progress during evaluation (WebSocket)
- Bulk actions (delete multiple runs)
- Custom test case management UI
