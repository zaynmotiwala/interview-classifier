# Email Classifier Interview Project

This is a simplified, standalone version of the RevReply Email Classifier system, created for technical interview assessment.

## Project Overview

This project demonstrates a production-ready AI-powered email classification system using:
- **FastAPI** for the REST API
- **DSPy** for LLM orchestration and prompt optimization
- **Azure OpenAI (GPT-5-mini)** for inference
- Pre-optimized classifier model using MIPRO (Multi-prompt Instruction Proposal Optimizer)

## What This System Does

The classifier analyzes email conversation threads and determines the prospect's intent. It classifies emails into 14 categories:
- `INTERESTED` - Prospect shows interest
- `INFORMATION REQUEST` - Asking questions
- `NOT INTERESTED` - Clear rejection
- `CHECK BACK LATER` - Not ready now, check later
- `SOFT OBJECTION` - Hesitation or concern
- `MEETING REQUEST` - Requesting to schedule a meeting
- `MEETING BOOKED` - Meeting confirmed
- `DELEGATED` - Forwarded to someone else
- `DO NOT CONTACT` - Explicit request to stop contact
- `OUT OF OFFICE` - Automated OOO response
- `WRONG PERSON` - Misdirected email
- `REQUEST COMPLETE` - Task completed, no further action
- `AUTOMATIC REPLY` - System-generated response
- `THREAD COMPLETE` - Conversation naturally concluded

## Setup Instructions

### 1. Install Dependencies

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 2. Configure Environment Variables

Copy the `.env.example` file to `.env` and fill in your Azure OpenAI credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your Azure OpenAI credentials:
```
AZURE_API_KEY=your_azure_api_key_here
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-08-01-preview
ENVIRONMENT=local
API_KEY=test-api-key-12345
```

### 3. Run the Server

```bash
poetry run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Classify an email
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "email_thread": "Hi, I would love to learn more about your product. Can we schedule a call next week?",
    "thread_id": "test-001"
  }'
```

### 5. Run Tests

```bash
poetry run pytest -v
```

## Project Structure

```
interview-classifier-project/
├── app/
│   ├── main.py                          # FastAPI application & /api/classify endpoint
│   ├── dual_logger.py                   # Logging utilities
│   ├── version.py                       # Version info
│   └── classes/
│       ├── modules/
│       │   └── classification/
│       │       ├── classifier_optimized.py    # Main classifier module
│       │       └── revclassifier-optimized-*/ # Pre-compiled MIPRO model
│       ├── signatures/
│       │   ├── classifier_optimized.py        # DSPy signature definition
│       │   ├── delegation_detector.py         # OOO delegation detection
│       │   ├── phone_number_extractor.py      # Phone number extraction
│       │   └── team_alerter.py                # Same-day call detection
│       └── models/
│           └── revreplyclassifier.py          # Pydantic request/response models
├── tests/
│   ├── test_classifier.py               # Basic classifier tests
│   └── conftest.py                      # Pytest fixtures
├── pyproject.toml                       # Poetry dependencies
├── .env.example                         # Environment template
└── README.md                            # This file
```

## Interview Tasks & Assessment Areas

This project is designed to assess your ability to:

### 1. Code Comprehension
- Understand the DSPy framework and how it differs from raw LLM calls
- Trace the flow from API endpoint → classifier module → DSPy signature
- Understand the rule-based override system and its purpose

### 2. System Design
- Evaluate the multi-checkpoint override architecture
- Assess the use of pre-compiled models vs. runtime optimization
- Understand token usage tracking and logging design

### 3. Production Readiness
- Identify potential issues or edge cases
- Suggest improvements for reliability, performance, or maintainability
- Consider monitoring, error handling, and debugging approaches

### 4. Code Quality
- Review code organization and separation of concerns
- Evaluate error handling patterns
- Assess test coverage and testing strategy

## Key Areas to Explore

1. **Rule-Based Overrides** (lines 180-542 in `classifier_optimized.py`)
   - Why do we have rule-based overrides on top of an LLM classifier?
   - Can you identify edge cases or conflicts between rules?
   - How would you improve or refactor this system?

2. **Token Usage Tracking** (lines 656-714 in `classifier_optimized.py`)
   - How does the system track token usage across multiple LLM calls?
   - What happens if delegation detection or phone extraction is enabled?
   - Is the current implementation correct?

3. **Feature Flags** (`ooo_toggle`, `check_same_day_phone_request`)
   - How do these features integrate with the main classification flow?
   - What are the performance implications?
   - How would you test these features?

4. **Error Handling**
   - What happens if the classifier model fails to load?
   - How are LLM API failures handled?
   - What monitoring would you add?

## Discussion Questions

1. What improvements would you make to this codebase?
2. How would you refactor the override logic to be more maintainable?
3. What performance optimizations could be applied?
4. How would you add telemetry/observability to this system?
5. What additional tests would you write?
6. If you had to add a new classification category, how would you do it?

## Technical Deep Dives

### DSPy Framework
This project uses DSPy 3.1.0, which provides:
- **Signatures**: Define input/output specifications for LLM tasks
- **Modules**: Composable LLM programs with optimization support
- **MIPRO Optimizer**: Multi-prompt instruction optimization for better accuracy
- **Caching**: Built-in prompt caching for cost/latency optimization

### Pre-Compiled Model
The `revclassifier-optimized-*` directory contains a MIPRO-optimized classifier that was trained on production data. This means:
- Instructions and few-shot examples are pre-tuned
- Better accuracy than raw prompts
- Faster inference (no runtime optimization needed)
- Model can be loaded with `dspy.load()`

### Azure OpenAI GPT-5-mini
This project uses GPT-5-mini with specific settings:
- `temperature=1.0` (required for GPT-5-mini)
- `reasoning_effort="medium"` (reduces thinking tokens, improves latency)
- `max_tokens=16000` (supports long email threads)

## License

This is a sample project for interview purposes only.
