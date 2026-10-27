# Interview Project Handoff

## What This Is

A fully-functional, standalone email classification system extracted from the RevReply PromptEngine project. This is designed to be used as a technical interview assessment for head of engineering candidates.

## What's Included

### Core Application (Fully Runnable)
- ✅ FastAPI application with `/api/classify` endpoint
- ✅ Pre-compiled DSPy MIPRO-optimized classifier model
- ✅ Request/response models with Pydantic validation
- ✅ API key authentication
- ✅ Health check endpoint
- ✅ Comprehensive logging and error handling
- ✅ Token usage tracking

### Documentation
- ✅ **README.md** - Project overview, architecture, key concepts
- ✅ **QUICKSTART.md** - Setup instructions, common issues, API reference
- ✅ **INTERVIEW_GUIDE.md** - Interview questions, scoring rubric, assessment criteria
- ✅ **HANDOFF.md** - This file

### Testing
- ✅ Basic classification tests
- ✅ Rule-based override tests
- ✅ Context-awareness tests
- ✅ Token usage tests
- ✅ Shell script for manual API testing (`test_api.sh`)

### Configuration
- ✅ `.env.example` - Environment variable template
- ✅ `pyproject.toml` - Poetry dependencies (minimal)
- ✅ `.gitignore` - Standard Python gitignore

## File Structure

```
interview-classifier-project/
├── Documentation
│   ├── README.md              # Main project overview
│   ├── QUICKSTART.md          # Setup guide
│   ├── INTERVIEW_GUIDE.md     # Interview assessment guide
│   └── HANDOFF.md             # This file
│
├── Application Code
│   └── app/
│       ├── main.py                           # FastAPI app
│       ├── version.py, dual_logger.py, utils.py
│       └── classes/
│           ├── modules/classification/
│           │   ├── classifier_optimized.py            # Main classifier (700+ lines)
│           │   └── revclassifier-optimized-*/         # Pre-compiled model
│           ├── signatures/                            # DSPy signatures
│           │   ├── classifier_optimized.py
│           │   ├── delegation_detector.py
│           │   ├── phone_number_extractor.py
│           │   └── team_alerter.py
│           ├── models/
│           │   └── revreplyclassifier.py              # Request/response models
│           └── helpers/
│               └── token_calculator.py
│
├── Tests
│   └── tests/
│       ├── conftest.py
│       └── test_classifier.py
│
└── Configuration
    ├── pyproject.toml         # Poetry dependencies
    ├── .env.example           # Environment template
    ├── .env                   # Actual credentials (copy from PromptEngine)
    ├── .gitignore
    └── test_api.sh            # API test script
```

## How to Use for Interviews

### Before the Interview (15 min setup)

1. **Verify it runs:**
   ```bash
   cd interview-classifier-project
   poetry install
   poetry run uvicorn app.main:app --reload
   ```

2. **Test basic functionality:**
   ```bash
   ./test_api.sh
   ```

3. **Review INTERVIEW_GUIDE.md** - familiarize yourself with questions and scoring

4. **Prepare the environment:**
   - Clone this project to a fresh directory
   - Remove `.env` (candidate should configure it)
   - Optional: Remove `.git` if you want a clean slate

### During the Interview (60-90 min)

**Part 1: Code Exploration (30 min)**
- Give candidate the project
- Ask them to set it up and explore the codebase
- Have them walk through the request flow
- See INTERVIEW_GUIDE.md for detailed questions

**Part 2: System Design (20 min)**
- Discuss scaling to 100K requests/day
- Cost optimization strategies
- Reliability and monitoring

**Part 3: Deep Dive - Rule Overrides (15 min)**
- Why do we need rule-based overrides?
- How would you refactor the override logic?
- What edge cases do you see?

**Part 4: Production Readiness (10 min)**
- Deployment strategy
- Observability and debugging
- Security considerations

**Part 5: Hands-On (15-20 min, optional)**
- Add a feature: detect document requests
- Write a test for it
- Evaluate code quality and execution speed

### After the Interview

Use the scoring rubric in INTERVIEW_GUIDE.md to evaluate:
- Technical skills (40 pts)
- AI/ML engineering (30 pts)
- Leadership & communication (30 pts)
- Bonus: Hands-on task (10 pts)

## Key Discussion Points

### Why This Project is Good for Assessment

1. **Real Production Code**: Not a toy problem - this is simplified but real production code
2. **AI/ML Focus**: Tests understanding of LLM-specific challenges (cost, non-determinism, evaluation)
3. **System Design**: Not just coding - requires thinking about scale, reliability, observability
4. **Multiple Dimensions**: Tests code comprehension, architecture, testing, production thinking
5. **Flexible Depth**: Can go shallow (explain the code) or deep (refactor, optimize, extend)

### What You're Looking For

**Strong Candidates Will:**
- Quickly understand the flow from endpoint → classifier → DSPy signature
- Identify LLM costs as the primary concern and propose caching/optimization
- Recognize rule-based overrides as deterministic guards for critical cases
- Propose concrete refactoring (rule engine, separate modules) with trade-offs
- Think about monitoring, observability, and how to measure AI system quality
- Ask insightful questions about business context and user needs

**Weak Candidates Will:**
- Get lost in implementation details without understanding the big picture
- Not understand why we use DSPy vs. raw OpenAI calls
- Suggest "just let the LLM handle everything" without considering cost/reliability
- Propose over-engineered solutions without understanding the problem
- Not think about production concerns (monitoring, debugging, security)
- Can't complete the hands-on task in a reasonable time

## Customization Options

You can adjust the difficulty based on candidate level:

**For Senior Engineers (not leadership):**
- Skip system design discussion
- Focus more on code quality and testing
- Extend hands-on task (implement, not just design)

**For Engineering Managers:**
- More emphasis on system design and team collaboration
- Less emphasis on coding
- Ask about how they'd mentor team on AI/ML engineering

**For CTO/VP-level:**
- Focus on business impact, not implementation details
- Discuss build vs. buy decisions
- Team structure and hiring strategy

## Notes

### Environment Setup
- Candidate needs Azure OpenAI API credentials
- If they don't have access, provide a test endpoint or use your credentials
- Alternatively, modify the code to use OpenAI API directly (easier to get access)

### Time Estimates
- Code exploration: 15-20 min
- Part 1 (comprehension): 30 min
- Part 2 (system design): 20 min
- Part 3 (deep dive): 15 min
- Part 4 (production): 10 min
- Part 5 (hands-on): 15-20 min
- **Total: 60-90 min** (adjust based on depth)

### What's Simplified

This project is a simplified version of the full PromptEngine:
- No Elasticsearch, Supabase, or external dependencies
- Simplified logging (stdout only, not Elasticsearch)
- No RAG system, embeddings, or email history
- Single classifier endpoint (not 24 endpoints)
- Minimal error handling compared to production

**But it's still realistic:**
- Real DSPy code with MIPRO-optimized model
- Real rule-based override logic from production
- Real token usage tracking
- Real API structure

## Troubleshooting

### Candidate Can't Get It Running

**Common Issues:**
1. Poetry not installed → `curl -sSL https://install.python-poetry.org | python3 -`
2. Python version wrong → Needs Python 3.12+
3. Azure credentials invalid → Provide test credentials or use OpenAI directly
4. Classifier model missing → Verify `revclassifier-optimized-*` directory exists

**If all else fails:**
- Skip setup, just review the code
- Focus on system design and discussion
- Use API docs (`/docs` endpoint) to explain endpoints

### Interview Running Long

**Cut these parts first:**
1. Hands-on coding task (Part 5)
2. Production readiness discussion (Part 4)
3. Deep dive on rule overrides (Part 3)

**Keep these parts:**
- Code exploration and comprehension
- System design discussion

### Candidate Struggles with DSPy

**That's OK!** DSPy is relatively new. What matters is:
- Can they learn new frameworks quickly?
- Do they understand the value proposition (optimization, composability)?
- Can they adapt LLM engineering concepts (prompting, few-shot learning)?

Don't penalize for not knowing DSPy specifically. Look for learning ability and LLM understanding.

## Success Criteria

After using this project for interviews, you should be able to answer:

1. **Technical Depth**: Can they understand and explain complex code?
2. **AI/ML Readiness**: Do they understand LLM-specific engineering challenges?
3. **System Design**: Can they design scalable, cost-effective systems?
4. **Execution**: Can they ship code (hands-on task)?
5. **Leadership**: Can they mentor engineers and make technical decisions?

If yes to 4-5 of these → Strong hire
If yes to 2-3 → Acceptable hire
If yes to 0-1 → Pass

## Files You Can Share

**Before Interview:**
- Send QUICKSTART.md and .env.example
- Optionally send README.md for context

**During Interview:**
- Share entire project directory
- Give access to Azure OpenAI or provide test endpoint

**After Interview:**
- Nothing (this is proprietary code)

## Contact

Questions about this interview project? Contact the RevReply engineering team.

---

**Version:** 1.0.0
**Created:** 2026-05-02
**Last Updated:** 2026-05-02
