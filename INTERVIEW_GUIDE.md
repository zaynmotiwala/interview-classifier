# Interview Assessment Guide

This guide provides structured questions and assessment criteria for evaluating the head of engineering candidate using this sample project.

## Interview Structure (60-90 minutes)

### Part 1: Code Review & Comprehension (30 minutes)

**Setup:**
- Give candidate 15-20 minutes to explore the codebase
- Ask them to focus on understanding the flow from API endpoint to classification result
- They can run the server and tests if they want

**Discussion Questions:**

1. **Architecture Understanding**
   - "Can you walk me through what happens when a classification request comes in?"
   - "How does the DSPy framework differ from making raw OpenAI API calls?"
   - "Why do you think we use a pre-compiled classifier model instead of optimizing at runtime?"

2. **Code Quality Assessment**
   - "What do you think about the overall code organization?"
   - "What patterns or practices stand out to you (good or bad)?"
   - "How would you rate the error handling in this codebase?"

3. **Testing Strategy**
   - "Look at the test file - what's missing?"
   - "How would you improve the test coverage?"
   - "What would you add for integration testing?"

**What to Look For:**
- ✅ Can trace request flow accurately
- ✅ Understands DSPy's value proposition (optimization, composability)
- ✅ Identifies concrete improvement opportunities
- ✅ Thinks about edge cases and failure modes
- ❌ Gets lost in implementation details without understanding big picture
- ❌ Can't articulate what the code is doing
- ❌ Doesn't identify any issues or improvements

---

### Part 2: System Design Discussion (20 minutes)

**Scenario:** "This classifier currently handles ~10K requests/day. We're scaling to 100K/day and need to ensure reliability and cost-efficiency."

**Questions:**

1. **Performance & Scalability**
   - "What bottlenecks do you see in the current implementation?"
   - "How would you improve response time?"
   - "What caching strategies would you consider?"

2. **Cost Optimization**
   - "LLM API costs are our biggest expense. How would you reduce them?"
   - "What trade-offs would you consider between accuracy and cost?"
   - "Look at the rule-based overrides (lines 180-542 in classifier_optimized.py) - what's their purpose?"

3. **Reliability & Monitoring**
   - "How would you detect when the classifier is making bad predictions?"
   - "What metrics would you track?"
   - "How would you handle Azure OpenAI API failures or rate limits?"

**What to Look For:**
- ✅ Identifies LLM calls as the primary bottleneck
- ✅ Understands rule-based overrides save API calls and improve reliability
- ✅ Proposes caching strategies (prompt caching, response caching)
- ✅ Thinks about monitoring/alerting (accuracy metrics, latency p95/p99, error rates)
- ✅ Considers graceful degradation and fallback strategies
- ❌ Focuses only on traditional scaling (more servers) without considering LLM-specific concerns
- ❌ Doesn't understand cost implications of LLM usage
- ❌ No concrete ideas for monitoring AI system quality

---

### Part 3: Deep Dive - Rule-Based Overrides (15 minutes)

**Context:** The classifier has extensive rule-based override logic (lines 180-542 in `classifier_optimized.py`). These rules override the LLM's classification based on patterns.

**Questions:**

1. **Override Logic Analysis**
   - "Why do we need rule-based overrides on top of an LLM classifier?"
   - "Look at the 'data sourcing question' override (lines 241-262) - why is this a DO NOT CONTACT instead of letting the LLM decide?"
   - "Look at the THREAD COMPLETE override (lines 269-414) - can you spot any edge cases or conflicts?"

2. **Maintainability**
   - "This override logic is currently in one 700+ line file. How would you refactor it?"
   - "How would you make it easier to add/modify/remove rules?"
   - "How would you test these rules?"

3. **Evolution**
   - "If you had to add a new classification category, what would you change?"
   - "How would you balance rule-based overrides vs. improving the LLM prompts?"

**What to Look For:**
- ✅ Understands rules provide deterministic behavior for critical cases (legal/compliance)
- ✅ Recognizes rules save API costs by short-circuiting LLM calls
- ✅ Identifies maintainability concerns (long function, hard to test)
- ✅ Proposes concrete refactoring (rule engine, separate modules, decision tree)
- ✅ Thinks about testing strategies (parameterized tests, rule fixtures)
- ❌ Suggests "just let the LLM handle everything" without understanding risks
- ❌ Can't articulate refactoring approach
- ❌ Doesn't see edge cases or conflicts in the override logic

---

### Part 4: Production Readiness (10 minutes)

**Questions:**

1. **Deployment**
   - "How would you deploy this service to production?"
   - "What would you add before launching this?"
   - "How would you handle model updates (new classifier versions)?"

2. **Observability**
   - "What would you add for debugging production issues?"
   - "How would you trace a classification request through the system?"
   - "What alerts would you set up?"

3. **Security**
   - "What security concerns do you see?"
   - "The API key is validated at the endpoint level - is this sufficient?"
   - "What about PII in email threads?"

**What to Look For:**
- ✅ Proposes structured logging, tracing (OpenTelemetry), metrics
- ✅ Thinks about model versioning and gradual rollout (canary, A/B testing)
- ✅ Identifies PII concerns, suggests data sanitization or redaction
- ✅ Considers rate limiting, request validation, API key rotation
- ✅ Mentions containerization (Docker), orchestration (K8s), health checks
- ❌ Hasn't thought about observability in production AI systems
- ❌ Doesn't consider gradual rollout or rollback strategies
- ❌ Misses obvious security issues (API keys in code, PII logging)

---

### Part 5: Hands-On Task (15-20 minutes, optional)

**Task:** "Add a new feature: detect when a prospect requests a specific document (whitepaper, case study, pricing sheet) and return that in the API response."

**Acceptance Criteria:**
- New field in response: `document_requested: str | None`
- Must handle variations: "send me your pricing", "do you have a case study", "whitepaper?"
- Write at least one test

**What to Look For:**
- ✅ Asks clarifying questions (what types of documents? how precise?)
- ✅ Considers where to implement (DSPy signature? post-processing? new module?)
- ✅ Writes clean, testable code
- ✅ Adds tests before or alongside implementation
- ✅ Can articulate trade-offs (regex vs. LLM extraction)
- ❌ Jumps into coding without understanding requirements
- ❌ Produces messy, untested code
- ❌ Can't complete in reasonable time (suggests may struggle with execution)

---

## Scoring Rubric

### Technical Skills (40 points)
- **Code Comprehension** (10 pts): Can understand and explain complex code
- **System Design** (15 pts): Proposes scalable, cost-effective solutions
- **Problem Solving** (15 pts): Identifies issues and provides concrete solutions

### AI/ML Engineering (30 points)
- **LLM Understanding** (10 pts): Understands LLM APIs, prompting, DSPy framework
- **AI System Design** (10 pts): Monitoring, evaluation, versioning for AI systems
- **Cost/Quality Trade-offs** (10 pts): Balances accuracy, latency, cost

### Leadership & Communication (30 points)
- **Communication** (10 pts): Explains technical concepts clearly
- **Critical Thinking** (10 pts): Asks good questions, challenges assumptions
- **Pragmatism** (10 pts): Proposes practical solutions, not over-engineering

### Bonus Points (10 points)
- **Hands-On Task** (10 pts): Completes coding task efficiently and cleanly

**Total: 110 points**

**Scoring:**
- 90-110: Excellent - Strong hire
- 70-89: Good - Hire with minor reservations
- 50-69: Mixed - Needs further discussion
- Below 50: Weak - Do not hire

---

## Red Flags to Watch For

1. **Over-Engineering**: Proposes complex solutions without understanding the problem
2. **Analysis Paralysis**: Can't make decisions or propose concrete solutions
3. **No AI/ML Experience**: Doesn't understand LLM-specific challenges (cost, non-determinism, evaluation)
4. **Poor Communication**: Can't explain technical concepts clearly
5. **Defensive**: Gets defensive when code quality is questioned
6. **No Curiosity**: Doesn't ask questions or dig deeper
7. **Execution Issues**: Can't complete simple coding task in reasonable time

---

## Green Flags to Look For

1. **Systems Thinking**: Connects code to business impact (cost, reliability, user experience)
2. **Pragmatic**: Proposes simple solutions first, then sophisticated ones
3. **AI-Aware**: Understands unique challenges of building with LLMs
4. **Ownership Mindset**: Thinks about production, not just features
5. **Collaborative**: Asks questions, builds on your ideas
6. **Growth Mindset**: Admits gaps in knowledge, shows interest in learning
7. **Execution Focused**: Delivers working code quickly in hands-on task

---

## Follow-Up Questions

Based on their responses, here are some good follow-up questions:

**If they propose caching:**
- "How would you handle cache invalidation?"
- "What would you cache - requests, responses, or LLM outputs?"

**If they mention monitoring:**
- "How do you measure classification accuracy in production when you don't have ground truth?"
- "What would you do if accuracy suddenly drops?"

**If they suggest refactoring:**
- "Show me how you'd refactor one of the override rules."
- "How would you ensure the refactoring doesn't break existing behavior?"

**If they mention tests:**
- "How would you test non-deterministic LLM outputs?"
- "What's your philosophy on mocking vs. integration tests for AI systems?"

---

## Post-Interview Evaluation

After the interview, discuss with your team:

1. **Technical Depth**: Do they have the skills to lead a technical team?
2. **AI/ML Readiness**: Can they build and scale AI-powered products?
3. **Execution**: Can they ship features and not just design them?
4. **Leadership Potential**: Can they mentor engineers and drive technical decisions?
5. **Culture Fit**: Do they align with your engineering values?

Remember: You're hiring a *head of engineering*, not a senior IC. Look for leadership qualities, not just coding skills.
