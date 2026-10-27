# Quick Start Guide

Get the interview project running in 5 minutes.

## Prerequisites

- Python 3.12+
- Poetry (install: `curl -sSL https://install.python-poetry.org | python3 -`)
- Azure OpenAI API credentials

## Setup

### 1. Install Dependencies

```bash
cd interview-classifier-project
poetry install
```

This will install:
- FastAPI & Uvicorn (web framework)
- DSPy 3.1.0 (LLM orchestration)
- Pydantic (data validation)
- pytest (testing)

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Azure OpenAI credentials:

```bash
# Required
AZURE_API_KEY=your_azure_api_key
AZURE_API_BASE=https://your-resource.cognitiveservices.azure.com/
AZURE_API_VERSION=2024-08-01-preview

# Optional (defaults shown)
ENVIRONMENT=local
API_KEY=test-api-key-12345
```

### 3. Run the Server

```bash
poetry run uvicorn app.main:app --reload
```

You should see:
```
🚀 Starting Email Classifier API...
Version: 1.0.0
Environment: local
✅ Azure OpenAI LM initialized
✅ Classifier loaded successfully
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 4. Test the API

**Option A: Using curl**
```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "email_thread": "Hi, I would love to learn more about your product!",
    "thread_id": "test-001"
  }'
```

**Option B: Using the test script**
```bash
./test_api.sh
```

**Option C: Interactive API docs**
Open http://localhost:8000/docs in your browser

### 5. Run Tests

```bash
poetry run pytest -v
```

## Project Structure

```
interview-classifier-project/
├── app/
│   ├── main.py                    # FastAPI app & /api/classify endpoint
│   ├── version.py
│   ├── dual_logger.py
│   ├── utils.py
│   └── classes/
│       ├── modules/
│       │   └── classification/
│       │       ├── classifier_optimized.py     # Main classifier logic
│       │       └── revclassifier-optimized-*/  # Pre-compiled model
│       ├── signatures/
│       │   ├── classifier_optimized.py         # DSPy signature
│       │   ├── delegation_detector.py
│       │   ├── phone_number_extractor.py
│       │   └── team_alerter.py
│       ├── models/
│       │   └── revreplyclassifier.py          # Request/response models
│       └── helpers/
│           └── token_calculator.py
├── tests/
│   ├── conftest.py
│   └── test_classifier.py
├── README.md
├── INTERVIEW_GUIDE.md
├── QUICKSTART.md (this file)
├── pyproject.toml
├── .env.example
└── test_api.sh
```

## Common Issues

### "Classifier not available"

**Problem:** Server starts but returns 503 errors

**Solution:** Check that:
1. `.env` has valid Azure OpenAI credentials
2. `revclassifier-optimized-*` directory exists in `app/classes/modules/classification/`
3. No errors in server logs during startup

### "Module not found" errors

**Problem:** Import errors when running tests or server

**Solution:**
```bash
# Reinstall dependencies
poetry install

# Or activate the virtual environment
poetry shell
```

### Tests fail with Azure API errors

**Problem:** Tests fail with 401/403 errors

**Solution:** Verify Azure credentials:
```bash
poetry run python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key:', os.getenv('AZURE_API_KEY')[:10] + '...')
print('API Base:', os.getenv('AZURE_API_BASE'))
"
```

## API Endpoints

### `GET /`
Root endpoint with API info

### `GET /health`
Health check - returns classifier status

### `POST /api/classify`
Classify email thread

**Request:**
```json
{
  "email_thread": "Email text...",
  "thread_id": "optional-id",
  "user_id": 123,
  "persona_id": 456,
  "ooo_toggle": false,
  "check_same_day_phone_request": false,
  "disqualification_criteria": ""
}
```

**Response:**
```json
{
  "prospect_classification": "INTERESTED",
  "reasoning": "Explanation...",
  "lm_usage": {
    "user_id": 123,
    "persona_id": 456,
    "classifier": {
      "completion_tokens": 50,
      "prompt_tokens": 200,
      "total_tokens": 250
    }
  },
  "total_tokens": {
    "input": 200,
    "output": 50
  },
  "phone_number": "",
  "same_day_phone_request_detected": false
}
```

## Next Steps

- Read [README.md](./README.md) for project overview
- Review [INTERVIEW_GUIDE.md](./INTERVIEW_GUIDE.md) for assessment criteria
- Explore the code starting from `app/main.py`
- Run tests with `pytest -v`
- Check the interactive API docs at http://localhost:8000/docs

## Need Help?

- Check server logs for errors
- Verify environment variables in `.env`
- Ensure Poetry installed dependencies correctly
- Test Azure OpenAI connection separately
