# ⚠️ URGENT: Dependency Fix Required

## Issue

Both the interview project AND the parent PromptEngine project have a broken `litellm` dependency (transitive dependency from DSPy 3.1.0).

**Error:**
```
ModuleNotFoundError: No module named 'litellm.types'
```

## Quick Fix (Run This First)

### In the main PromptEngine project:

```bash
cd c:/Github/PromptEngine

# Remove broken litellm
poetry run pip uninstall -y litellm

# Install working version
poetry run pip install litellm==1.62.9

# Verify fix
poetry run python -c "import dspy; print('✅ DSPy works!')"
```

### Then for interview project:

```bash
cd c:/tmp/interview-classifier-project

# Copy the fixed dependency
poetry env use $(cd ../Github/PromptEngine && poetry env info --path)/bin/python

# OR manually fix
poetry run pip uninstall -y litellm
poetry run pip install litellm==1.62.9
```

## Verify Setup

```bash
cd c:/tmp/interview-classifier-project

# Test imports
poetry run python -c "import fastapi; import dspy; print('✅ All imports work!')"

# Run server
poetry run uvicorn app.main:app --reload
```

## Root Cause

DSPy 3.1.0 depends on `litellm` but doesn't pin the version. Recent litellm releases (1.63.0+) reorganized internal modules, breaking backwards compatibility.

## Permanent Fix

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
litellm = "1.62.9"  # Pin to last working version
```

Then:
```bash
poetry lock --no-update
poetry install
```

## If Interview is Imminent

**Option 1: Pre-fix everything (5 min)**
- Run the quick fix above
- Verify server starts
- Give candidate a working environment

**Option 2: Skip running code**
- Focus on code review and discussion
- Use `/docs` API documentation
- Don't waste interview time debugging dependencies

**Option 3: Include debug as assessment**
- Give candidate the broken environment
- See how they debug dependency issues
- Provide hint if stuck > 10 minutes

## Status Check

Run this to verify everything works:

```bash
#!/bin/bash
echo "Checking PromptEngine..."
cd c:/Github/PromptEngine
poetry run python -c "import dspy" && echo "✅ PromptEngine OK" || echo "❌ PromptEngine BROKEN"

echo ""
echo "Checking interview project..."
cd c:/tmp/interview-classifier-project
poetry run python -c "import dspy" && echo "✅ Interview project OK" || echo "❌ Interview project BROKEN"

echo ""
echo "Starting server test..."
cd c:/tmp/interview-classifier-project
timeout 5 poetry run uvicorn app.main:app 2>&1 | grep -i "started\|error" || echo "Server test failed"
```

Save as `check_status.sh` and run: `bash check_status.sh`

---

**Priority:** HIGH  
**Time to fix:** 5 minutes  
**Must fix before:** Interview
