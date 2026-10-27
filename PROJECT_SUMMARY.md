# Interview Classifier Project - Complete Summary

## ✅ Project Created Successfully

A fully-functional, standalone email classification system extracted from RevReply's PromptEngine, designed for head of engineering interviews.

**Location:** `c:\tmp\interview-classifier-project\`

---

## 📦 What's Included

### Application Code
- ✅ FastAPI application with `/api/classify` endpoint
- ✅ Pre-compiled DSPy MIPRO-optimized classifier model
- ✅ Complete rule-based override system (700+ lines of production logic)
- ✅ Request/response models with Pydantic
- ✅ API key authentication
- ✅ Token usage tracking
- ✅ Comprehensive logging

### Documentation (7 files)
1. **START_HERE.md** - Quick reference guide (read this first!)
2. **QUICKSTART.md** - 5-minute setup guide
3. **README.md** - Full project overview and architecture
4. **INTERVIEW_GUIDE.md** - Interview questions, scoring rubric, assessment criteria
5. **HANDOFF.md** - How to use this for interviews
6. **SETUP_NOTES.md** - Environment setup details
7. **URGENT_SETUP_FIX.md** - Dependency issue fix (litellm)

### Tests
- ✅ 15 test cases covering basic classification, rule overrides, context-awareness
- ✅ Token usage verification tests
- ✅ Shell script for manual API testing

### Configuration
- ✅ `pyproject.toml` - Poetry dependencies
- ✅ `.env.example` - Environment variable template
- ✅ `.env` - Actual credentials (copied from PromptEngine)
- ✅ `.gitignore` - Standard Python gitignore

---

## 🎯 For the Interview

### Before Interview (Quick Setup)

```bash
# Navigate to project
cd c:\tmp\interview-classifier-project

# Option 1: Use parent environment (RECOMMENDED - avoids dependency issues)
cd c:\Github\PromptEngine && poetry shell
cd c:\tmp\interview-classifier-project
uvicorn app.main:app --reload

# Option 2: Standalone environment (may need dependency fix)
poetry install
poetry run uvicorn app.main:app --reload
```

### During Interview (60-90 min)

1. **Code Exploration** (30 min) - Candidate reviews and explains the code
2. **System Design** (20 min) - Scaling, cost optimization, reliability
3. **Deep Dive** (15 min) - Rule-based overrides and refactoring
4. **Production Readiness** (10 min) - Deployment, monitoring, security
5. **Hands-On Task** (15-20 min, optional) - Implement a feature

### Scoring (see INTERVIEW_GUIDE.md)
- **90-110 pts** = Strong hire
- **70-89 pts** = Hire with reservations
- **50-69 pts** = Mixed
- **<50 pts** = Pass

---

## 📁 Project Structure

```
interview-classifier-project/
├── Documentation/
│   ├── START_HERE.md              ← Read this first!
│   ├── QUICKSTART.md
│   ├── README.md
│   ├── INTERVIEW_GUIDE.md
│   ├── HANDOFF.md
│   ├── SETUP_NOTES.md
│   └── URGENT_SETUP_FIX.md
│
├── Application Code/
│   └── app/
│       ├── main.py                           # FastAPI application
│       ├── version.py, dual_logger.py, utils.py
│       └── classes/
│           ├── modules/classification/
│           │   ├── classifier_optimized.py            # Main classifier (755 lines)
│           │   └── revclassifier-optimized-*/         # Pre-compiled MIPRO model
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
├── Tests/
│   └── tests/
│       ├── conftest.py
│       └── test_classifier.py       # 15 test cases
│
└── Configuration/
    ├── pyproject.toml
    ├── .env.example
    ├── .env
    ├── .gitignore
    └── test_api.sh
```

---

## ⚠️ Known Issues & Fixes

### Issue: litellm Dependency Error

DSPy 3.1.0 has a transitive dependency on `litellm` which recently broke (versions 1.63.0+).

**Quick Fix (5 minutes):**

```bash
# In PromptEngine directory
cd c:\Github\PromptEngine
poetry run pip uninstall -y litellm
poetry run pip install 'litellm<1.63.0'

# Verify fix
poetry run python -c "import dspy; print('✅ Works!')"

# Then use this environment for interview project
poetry shell
cd c:\tmp\interview-classifier-project
uvicorn app.main:app --reload
```

**Alternative:** Use Option 1 from quick setup (parent environment).

See `URGENT_SETUP_FIX.md` for details.

---

## 🚀 Quick Test

After setup, verify everything works:

```bash
# Start server
uvicorn app.main:app --reload

# In another terminal, run tests
./test_api.sh

# Or manually test
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "classifier": "ok",
  "timestamp": "2026-05-02T..."
}
```

---

## 📊 Project Stats

- **Total Files:** 27 files
- **Python Files:** 21 files
- **Lines of Code:** ~1,500 (excluding tests)
- **Main Module:** 755 lines (classifier_optimized.py)
- **Documentation:** 2,000+ lines across 7 files
- **Test Cases:** 15
- **API Endpoints:** 3 (`/`, `/health`, `/api/classify`)
- **Classification Categories:** 14
- **Pre-compiled Model:** MIPRO-optimized DSPy classifier

---

## 🎓 What This Assesses

### For Head of Engineering Role

1. **Technical Depth**
   - Can understand complex AI/ML code
   - Traces request flow accurately
   - Identifies performance bottlenecks

2. **System Design**
   - Proposes scalable, cost-effective solutions
   - Thinks about monitoring and observability
   - Understands LLM-specific challenges

3. **Code Quality**
   - Identifies refactoring opportunities
   - Proposes concrete improvements
   - Balances pragmatism vs. perfectionism

4. **AI/ML Engineering**
   - Understands DSPy vs. raw LLM calls
   - Thinks about accuracy, cost, latency trade-offs
   - Considers non-determinism and testing strategies

5. **Leadership**
   - Communicates technical concepts clearly
   - Asks good questions
   - Shows ownership mindset

---

## ✅ Pre-Interview Checklist

- [ ] Fix litellm dependency (5 min)
- [ ] Verify server starts: `uvicorn app.main:app --reload`
- [ ] Test API: `./test_api.sh` or `curl http://localhost:8000/health`
- [ ] Review INTERVIEW_GUIDE.md (10 min)
- [ ] Prepare candidate environment (copy project or share screen)
- [ ] Have Azure OpenAI credentials ready (if candidate needs them)

---

## 📞 Quick Reference

**Start server:**
```bash
cd c:\Github\PromptEngine && poetry shell
cd c:\tmp\interview-classifier-project
uvicorn app.main:app --reload
```

**Run tests:**
```bash
poetry run pytest -v
```

**Test API:**
```bash
./test_api.sh
```

**API docs:**
http://localhost:8000/docs

---

## 🎯 Success Criteria

After interview, you should be able to answer:

1. ✅ Can they understand complex code?
2. ✅ Do they think about scale, cost, reliability?
3. ✅ Can they propose concrete improvements?
4. ✅ Do they ask good questions?
5. ✅ Can they execute (hands-on task)?

**If 4-5 yes → Strong hire**

---

## 📝 Next Steps

1. **Read START_HERE.md** - Quick overview
2. **Fix dependencies** - Run litellm fix (URGENT_SETUP_FIX.md)
3. **Test setup** - Verify server runs
4. **Review interview guide** - INTERVIEW_GUIDE.md
5. **Conduct interview** - 60-90 minutes
6. **Evaluate** - Use scoring rubric

---

## 📧 Questions?

- Setup issues → SETUP_NOTES.md
- Interview questions → INTERVIEW_GUIDE.md
- General info → README.md
- Quick start → QUICKSTART.md

---

**Created:** 2026-05-02  
**Version:** 1.0.0  
**Parent Project:** RevReply PromptEngine (c:\Github\PromptEngine)  
**Location:** c:\tmp\interview-classifier-project\

**Status:** ✅ Ready for interview (after dependency fix)
