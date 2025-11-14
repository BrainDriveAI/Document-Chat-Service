# Guide for AI Coding Agents

**Purpose:** Help AI assistants work effectively by learning from past decisions, failures, and discoveries.

---

## 🚀 Quick Start (Read These FIRST)

### Before Writing Any Code
✅ Check decisions: Does ADR exist for this approach?
✅ Check failures: Has this been tried and failed before?
✅ Check data-quirks: Any non-obvious data behavior?

**Commands to run:**
```bash
# Search for related decisions
grep -ri "keyword" docs/decisions/

# Search for past failures
grep -ri "keyword" docs/failures/

# Search for data quirks
grep -ri "keyword" docs/data-quirks/

# Search for integration patterns
grep -ri "keyword" docs/integrations/
```

---

## 📚 Knowledge Base Structure

```
docs/
├── decisions/           # Architecture Decision Records (ADRs)
├── failures/            # Lessons learned (what NOT to do)
├── data-quirks/         # Non-obvious data behavior
├── integrations/        # External system gotchas
└── AI-AGENT-GUIDE.md   # This file
```

---

## 🤖 Compounding Engineering: Your Role

**What is it?**
> You don't just write code - you **compound knowledge** for future developers/agents by documenting decisions, failures, and discoveries.

**Your mission:**
1. ✅ Write code (normal development)
2. ✅ **PLUS:** Document what you learned for the next developer/agent

**Think:**
"Six months from now, someone (human or AI) will work on this. What do they need to know to avoid my mistakes and build on my success?"

---

## 📝 When to Create Documentation (Auto-Compound)

### 1. Made an Architectural Decision? → Create ADR

**Trigger conditions:**
- ✅ Chose between 2+ implementation approaches
- ✅ Selected a library/framework
- ✅ Decided on data structure or pattern
- ✅ Changed core Clean Architecture layer interaction
- ✅ Chose design pattern (Strategy, Factory, etc.)
- ✅ Selected external service provider

**Action:**
```bash
cp docs/decisions/000-template.md docs/decisions/00X-your-decision.md
```

**Fill in:**
- **Context:** Why was this decision needed?
- **Problem:** What are you solving?
- **Decision:** What did you choose?
- **Consequences:** Pros/cons, tradeoffs
- **Alternatives:** What you rejected and WHY

**Examples:**
- Chose Ollama over OpenAI for embeddings → Create ADR
- Selected OptimizedHierarchicalChunking over SemanticChunking → Create ADR
- Decided to use Clean Architecture → Create ADR
- Chose ChromaDB over Qdrant → Create ADR

---

### 2. Discovered Data Behaves Weirdly? → Create Data Quirk Doc

**Trigger conditions:**
- ✅ Data format different than expected
- ✅ Table/collection has retention policy
- ✅ Field has NULL/invalid values unexpectedly
- ✅ Non-obvious relationships between entities
- ✅ Timezone/encoding inconsistencies
- ✅ API returns data in unexpected format

**Action:**
```bash
touch docs/data-quirks/00X-quirk-name.md
```

**Document:**
- **Behavior:** What's weird about the data?
- **Why it matters:** Impact on features/functionality
- **Root cause:** Why is it this way?
- **Detection:** How to identify this quirk
- **Correct patterns:** How to handle it properly

**Examples:**
- Ollama context window defaults to 2048 (not model max) → Document quirk
- ChromaDB embedding dimension must match exactly → Document quirk
- BM25 index requires manual refresh after updates → Document quirk
- mxbai-embed-large optimal batch size is 5-8 → Document quirk

---

### 3. Hit an Error or Made a Mistake? → Create Failure Log

**Trigger conditions:**
- ✅ Assumed something that was wrong
- ✅ Built feature that didn't work (later fixed)
- ✅ Used wrong approach (wasted >1 hour)
- ✅ Discovered anti-pattern
- ✅ Integration failed in unexpected way
- ✅ Performance issue not anticipated

**Action:**
```bash
cp docs/failures/000-template.md docs/failures/00X-failure-name.md
```

**Document:**
- **What happened:** The mistake/error
- **Root cause:** Why it happened
- **Impact:** Consequences (time wasted, bugs, etc.)
- **Lessons learned:** What NOT to do
- **Resolution:** How it was fixed
- **Prevention:** Checklist to avoid in future

**Examples:**
- Race condition in concurrent evaluation submissions → Document failure
- Ollama tests failed in CI (no Ollama instance) → Document failure
- Used incremental counting with async operations → Document failure
- Assumed embedding batch size of 32 would work (OOM) → Document failure

---

### 4. Integrated External System? → Create Integration Doc

**Trigger conditions:**
- ✅ Connected to new API/service
- ✅ Vendor-specific quirks discovered
- ✅ Authentication/authorization setup
- ✅ Error handling patterns established
- ✅ Rate limits/quotas encountered

**Action:**
```bash
touch docs/integrations/system-name.md
```

**Document:**
- **Purpose:** What does this integration do?
- **Authentication:** How to authenticate
- **Data format/schema:** Request/response structure
- **Quirks and gotchas:** Vendor-specific oddities
- **Error handling:** How to handle failures
- **Rate limits:** Throttling, quotas, retries

**Examples:**
- Ollama API integration → Document patterns
- Document Processor API integration → Document quirks
- ChromaDB integration → Document edge cases
- SQLite async patterns → Document best practices

---

## 🔍 Before Implementing Features

### Step 1: Search Existing Knowledge

**Always run these searches before implementing:**

```bash
# Check if decision already made
grep -ri "authentication" docs/decisions/
grep -ri "chunking" docs/decisions/

# Check for past failures
grep -ri "race condition" docs/failures/
grep -ri "timeout" docs/failures/

# Check for data quirks
grep -ri "embedding" docs/data-quirks/
grep -ri "context window" docs/data-quirks/

# Check integration patterns
grep -ri "ollama" docs/integrations/
```

### Step 2: Check with User if Uncertain

**If you're not sure:**
- ✅ ASK the user before building
- ❌ DON'T assume and waste time

**Questions to ask:**
- "I see ADR-003 chose ChromaDB. Should I use the same approach?"
- "Failure-001 shows concurrent counting failed. Should I re-query DB instead?"
- "Data-Quirk-001 says context window defaults to 2048. Should I set num_ctx explicitly?"

---

## 🔄 After You're Done

### Before Committing Code

**Checklist:**
- [ ] Did you make an architectural decision? → Create ADR
- [ ] Did you discover data quirk? → Document it
- [ ] Did you hit an error/mistake? → Create failure log
- [ ] Did you integrate external system? → Document patterns
- [ ] Did you learn something non-obvious? → Document it

### During Code Review

**Ask yourself:**
- "Will the next developer understand WHY I made this choice?"
- "If this fails in production, will logs point to the quirk documentation?"
- "Did I search docs/ before implementing, or reinvent the wheel?"

---

## 📋 Project-Specific Context

### This RAG Application

**Architecture:** Clean Architecture (Domain → Ports → Use Cases → Adapters → API)

**Key technologies:**
- **LLM/Embeddings:** Ollama (remote Cloud Run instances)
- **Vector Store:** ChromaDB
- **Search:** Hybrid (Vector + BM25 + Rank Fusion)
- **Database:** SQLite with async SQLAlchemy
- **Document Processing:** Remote spaCy Layout API

**Common patterns:**
- Use port interfaces (ABC/Protocol) for all external dependencies
- Services instantiated once at startup (`app.state`)
- Dependency injection via `app/api/deps.py`
- Background tasks for long-running operations
- Async operations for all I/O

**Known quirks:**
- Ollama context window defaults to 2048 (must set `num_ctx`)
- Embedding batch size 5-8 optimal for 16GB RAM
- ChromaDB telemetry must be disabled (`CHROMA_TELEMETRY=0`)
- BM25 index persists separately from vector store

### Search Before You Code

**Before implementing a use case:**
```bash
grep -ri "use case name" docs/decisions/
```

**Before choosing a library:**
```bash
grep -ri "library name" docs/decisions/
grep -ri "library name" docs/failures/
```

**Before implementing retrieval logic:**
```bash
grep -ri "retrieval" docs/decisions/
grep -ri "search" docs/data-quirks/
```

---

## 🎯 Your Goal as AI Agent

**Not just:** Write working code

**But also:** Leave knowledge for the next developer/agent

**Success metrics:**
1. ✅ Code works
2. ✅ Tests pass
3. ✅ **Future developer avoids your mistakes** (documented in failures/)
4. ✅ **Future developer understands your decisions** (documented in decisions/)
5. ✅ **Future developer handles edge cases** (documented in data-quirks/)

---

## 🚨 Anti-Patterns (DON'T DO THIS)

**❌ Don't:**
- Implement feature without checking docs/ first
- Make architectural decision without creating ADR
- Hit error >1hr and not document failure
- Discover quirk and not document it
- Assume previous developer had no reason for choice

**✅ Do:**
- Search docs/ before every implementation
- Create ADR for every non-trivial decision
- Document every failure that wasted time
- Document every surprising behavior
- Read existing ADRs to understand "why"

---

## 📖 Example Workflow

### Scenario: Add new chunking strategy

1. **Search first:**
```bash
grep -ri "chunking" docs/decisions/
# Found: ADR-007 chose OptimizedHierarchicalChunking
# Reason: Preserves document structure, better retrieval
```

2. **Ask user:**
"ADR-007 chose OptimizedHierarchicalChunking. Should I extend it or create new strategy?"

3. **Implement based on decision**

4. **Document:**
```bash
# If architectural change
cp docs/decisions/000-template.md docs/decisions/008-semantic-chunking-addition.md

# If discovered quirk
touch docs/data-quirks/003-semantic-chunking-sentence-boundary.md
```

---

## 🔗 Quick Reference

**Templates:**
- ADR: `docs/decisions/000-template.md`
- Failure: `docs/failures/000-template.md`

**Search commands:**
```bash
# Find all decisions about X
grep -ri "keyword" docs/decisions/

# Find all failures related to Y
grep -ri "keyword" docs/failures/

# Find all quirks about Z
grep -ri "keyword" docs/data-quirks/
```

**When in doubt:**
1. Search docs/
2. Ask user
3. Document your decision

---

That's compounding engineering. Every session makes the next one faster. 🚀
