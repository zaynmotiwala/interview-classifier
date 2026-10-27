# Setup Notes for Interviewers

## Important: Dependency Installation Issue

There's a known issue with DSPy 3.1.0's transitive dependency on `litellm` that can cause installation failures.

### Recommended Setup (Option 1): Use Parent Project Environment

The easiest approach is to run this project using the same Poetry environment as the main PromptEngine project:

```bash
# From the PromptEngine directory
cd c:/Github/PromptEngine
poetry install  # This already works

# Then run the interview project from here
cd c:/tmp/interview-classifier-project
poetry run --directory c:/Github/PromptEngine uvicorn app.main:app --reload
```

Or set up the Python path:

```bash
# Activate the PromptEngine environment
cd c:/Github/PromptEngine
poetry shell

# Navigate to interview project
cd c:/tmp/interview-classifier-project

# Run the server
uvicorn app.main:app --reload
```

### Alternative Setup (Option 2): Fix Dependencies Manually

If you want an isolated environment, add this constraint to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = ">=3.12,<3.13"
fastapi = "^0.115.11"
uvicorn = ">=0.23.1"
python-dotenv = "^1.0.1"
dspy = "3.1.0"
pydantic = "2.10.6"
httpx = ">=0.26.0,<0.29.0"
litellm = "1.62.9"  # Add this line to pin working version
```

Then:
```bash
poetry lock --no-update
poetry install
```

### Quick Verification

Test that imports work:

```bash
poetry run python -c "import fastapi; import dspy; print('✅ Imports successful!')"
```

### For the Interview

**Option A: Pre-setup (Recommended)**
- Set up the environment yourself before the interview
- Verify the server runs: `poetry run uvicorn app.main:app --reload`
- Test with `./test_api.sh`
- Give the candidate a working environment

**Option B: Setup as part of interview**
- Include environment setup as part of the assessment
- Observe how the candidate debugs dependency issues
- Provide hints if they get stuck (don't waste interview time)

**Option C: Skip running code**
- Focus on code review and discussion
- Use the API docs (`/docs` endpoint) for reference
- Have candidate explain what the code does without running it

## Pre-Interview Checklist

- [ ] Environment set up and tested
- [ ] Server starts successfully
- [ ] `/health` endpoint returns 200 OK
- [ ] `/api/classify` endpoint works with test data
- [ ] `./test_api.sh` passes all tests
- [ ] `.env` file has valid Azure OpenAI credentials
- [ ] Tests run: `poetry run pytest -v`
- [ ] INTERVIEW_GUIDE.md reviewed

## Environment Variables Needed

Copy these from the main PromptEngine `.env`:

```bash
# Required for classifier
AZURE_API_KEY=...
AZURE_API_BASE=...
AZURE_API_VERSION=2024-08-01-preview

# Optional
ENVIRONMENT=local
API_KEY=test-api-key-12345
```

## Quick Test Script

Create `verify_setup.py`:

```python
#!/usr/bin/env python3
"""Verify the interview project is set up correctly."""
import os
import sys

def check_env():
    """Check environment variables."""
    required = ['AZURE_API_KEY', 'AZURE_API_BASE', 'AZURE_API_VERSION']
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    else:
        print("✅ Environment variables configured")
        return True

def check_imports():
    """Check Python imports."""
    try:
        import fastapi
        import dspy
        import pydantic
        print("✅ Python imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def check_classifier_model():
    """Check classifier model exists."""
    model_dir = "app/classes/modules/classification"
    import os
    
    models = [d for d in os.listdir(model_dir) 
              if d.startswith("revclassifier-optimized-")]
    
    if models:
        print(f"✅ Classifier model found: {models[0]}")
        return True
    else:
        print("❌ No classifier model found")
        return False

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Verifying interview project setup...\n")
    
    checks = [
        check_env(),
        check_imports(),
        check_classifier_model()
    ]
    
    if all(checks):
        print("\n✅ All checks passed! Ready for interview.")
        sys.exit(0)
    else:
        print("\n❌ Setup incomplete. Fix issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run it:
```bash
poetry run python verify_setup.py
```

## Troubleshooting

### Server won't start

```bash
# Check logs
poetry run uvicorn app.main:app --reload --log-level debug

# Common issues:
# 1. .env file missing → cp .env.example .env
# 2. AZURE_API_KEY invalid → check credentials
# 3. Classifier model missing → verify revclassifier-optimized-* exists
```

### Tests fail

```bash
# Run with verbose output
poetry run pytest -v -s

# Common issues:
# 1. Azure API credentials → check .env
# 2. Network issues → check firewall/proxy
# 3. Rate limiting → wait and retry
```

### Import errors

```bash
# Verify environment
poetry env info

# Reinstall
poetry env remove python3.12
poetry install

# Or use parent project environment (recommended)
```

## Summary

**Easiest path:**
1. Use the main PromptEngine environment (already working)
2. Copy `.env` from PromptEngine
3. Test: `poetry run uvicorn app.main:app --reload`
4. Verify: `./test_api.sh`
5. Done!

**Time estimate:** 5-10 minutes for setup if using parent environment
