# Data Quirk: Git Commit Message Format Requirements

**Date:** 2025-11-14
**Category:** Development Tools
**Impact:** Medium

## Behavior

Git commit messages have standard length limits:
- **Subject line:** 50 characters recommended, 72 max
- **Body:** Should be separated from subject by blank line
- **Format:** `git commit -m "subject" -m "body"` for proper separation

## Why It Matters

**Impact on workflow:**
- Long subject lines break git log formatting
- Combined subject+body in single `-m` flag formats incorrectly
- Git tools (GitHub, GitLab) parse first line as PR title
- Violating conventions makes history hard to read

**Best practices:**
- Subject: Concise summary (<50 chars ideal, <72 max)
- Body: Detailed explanation (no length limit)
- Separate with blank line

## Root Cause

Git's design separates commit messages into:
1. Subject (first line) - summary for `git log --oneline`
2. Body (after blank line) - detailed description

Using single `-m` flag with multiline text doesn't create proper separation.

## Detection

**Symptoms:**
- `git log` shows malformed messages
- GitHub PR titles include body text
- Commit message validation hooks fail

**Check:**
```bash
git log --oneline  # Should show clean subject lines
git log -1         # Should show subject + blank line + body
```

## Correct Patterns

### ❌ Wrong (single -m with multiline)
```bash
git commit -m "$(cat <<'EOF'
Long subject with body text all together
This gets treated as one line
EOF
)"
```

### ✅ Correct (separate -m flags)
```bash
git commit -m "docs: add comprehensive documentation system" -m "$(cat <<'EOF'
Add Owner's Manual, Compounding Engineering knowledge base, and AI Agent Guide

## What's Added
...detailed body here...
EOF
)"
```

### ✅ Also Correct (interactive editor)
```bash
git commit
# Opens editor with proper formatting:
# Line 1: Subject
# Line 2: <blank>
# Line 3+: Body
```

### ✅ Best Practice Structure
```
<type>: <subject under 50 chars>

<blank line>
<detailed body explanation>
- Bullet points
- More details
- References

Closes #123
```

**Common types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `test:` - Tests
- `chore:` - Maintenance

## Example (Correct Format)

```bash
git commit -m "docs: add comprehensive documentation system" -m "Add Owner's Manual, Compounding Engineering knowledge base, and AI Agent Guide

## What's Added

### 1. Owner's Manual
- Complete user/operator guide
- Troubleshooting reference
- Performance tuning

### 2. Compounding Engineering
- ADRs, failure logs, quirks, integrations
- Templates for all doc types
- 2 example failure logs

### 3. AI Agent Guide
- Auto-documentation triggers
- Search-before-implement workflow

### 4. Enhanced FOR-AI-CODING-AGENTS.md
- Trigger conditions
- Pre-commit checklist

Closes #14"
```

## Prevention Checklist

Before committing:
- [ ] Subject line <50 chars (check with `echo -n "subject" | wc -m`)
- [ ] Use `-m "subject" -m "body"` format (not single `-m`)
- [ ] Include type prefix (`docs:`, `feat:`, `fix:`)
- [ ] Body explains why, not what
- [ ] Reference issue if applicable (`Closes #123`)

## Tools to Help

**Check subject length:**
```bash
echo -n "your commit subject here" | wc -m
# Output: character count (should be <50)
```

**Lint commits (optional):**
```bash
npm install -g @commitlint/cli @commitlint/config-conventional
echo "module.exports = {extends: ['@commitlint/config-conventional']}" > commitlint.config.js
```

## Related Documentation

- **Git best practices:** https://chris.beams.io/posts/git-commit/
- **Conventional Commits:** https://www.conventionalcommits.org/
- **GitHub PR titles:** First line of commit becomes PR title
