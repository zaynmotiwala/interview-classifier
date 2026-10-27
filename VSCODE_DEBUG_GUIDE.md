# VSCode Debugging Guide

## ✅ Setup Complete!

I've created VSCode configuration files for you:
- `.vscode/launch.json` - Debug configurations
- `.vscode/settings.json` - Python and editor settings
- `.vscode/tasks.json` - Quick tasks (run server, tests, etc.)

---

## 🚀 How to Debug the FastAPI Server

### Method 1: Using the Debug Panel (Recommended)

1. **Open the Debug Panel**
   - Click the "Run and Debug" icon in the left sidebar (or press `Ctrl+Shift+D`)
   - OR go to: View → Run

2. **Select Debug Configuration**
   - At the top of the debug panel, you'll see a dropdown
   - Select: **"FastAPI: Debug Server"**

3. **Set Breakpoints** (Optional but useful!)
   - Open `app/main.py`
   - Click in the left margin (gutter) next to line numbers to add red dots (breakpoints)
   - Good places to start:
     - Line 150 in `app/main.py` - Inside the `/api/classify` endpoint
     - Line 212 in `classifier_optimized.py` - Where classification happens

4. **Start Debugging**
   - Click the green **Play** button (▶️) or press `F5`
   - You should see:
     ```
     🚀 Starting Email Classifier API...
     ✅ Azure OpenAI LM initialized
     ✅ Classifier loaded successfully
     INFO:     Uvicorn running on http://127.0.0.1:8000
     ```

5. **Test the API**
   - Open a new terminal: `Ctrl+Shift+\``
   - Run: `./test_api.sh`
   - OR use curl:
     ```bash
     curl -X POST http://localhost:8000/api/classify \
       -H "Content-Type: application/json" \
       -H "X-API-Key: test-api-key-12345" \
       -d '{"email_thread": "I would love to learn more!", "thread_id": "test-001"}'
     ```

6. **When a Breakpoint Hits**
   - Execution will pause at your breakpoint
   - You'll see:
     - **Variables panel** - Inspect all variables in scope
     - **Call Stack** - See the execution path
     - **Watch panel** - Add expressions to monitor
     - **Debug Console** - Execute Python code in the current context

7. **Debug Controls** (Top toolbar)
   - **Continue** (`F5`) - Resume execution
   - **Step Over** (`F10`) - Execute current line, don't enter functions
   - **Step Into** (`F11`) - Enter the function being called
   - **Step Out** (`Shift+F11`) - Exit current function
   - **Restart** (`Ctrl+Shift+F5`) - Restart the server
   - **Stop** (`Shift+F5`) - Stop debugging

---

## 🧪 How to Debug Tests

### Debug All Tests

1. Open Debug Panel (`Ctrl+Shift+D`)
2. Select: **"Python: Run Tests"**
3. Press `F5` to start
4. Set breakpoints in `tests/test_classifier.py` if needed

### Debug a Single Test File

1. Open `tests/test_classifier.py`
2. Set a breakpoint inside a test function
3. Open Debug Panel
4. Select: **"Python: Run Single Test File"**
5. Press `F5`

### Alternative: Use VSCode Test Explorer

1. Click the **Testing** icon in the left sidebar (flask icon)
2. Click **"Configure Python Tests"** → Select **pytest**
3. You'll see all tests in a tree view
4. Right-click any test → **"Debug Test"**

---

## 📝 Debugging Tips & Tricks

### 1. Conditional Breakpoints

Right-click a breakpoint (red dot) → **Edit Breakpoint** → Add condition:
```python
request.thread_id == "test-001"
```
Now it only pauses when that condition is true!

### 2. Logpoints (Print without stopping)

Right-click in the gutter → **Add Logpoint**
```python
Classification: {result.classification}
```
This prints to Debug Console without stopping execution.

### 3. Debug Console

When paused at a breakpoint, use the Debug Console to:
```python
# Inspect variables
>>> result.classification
'INTERESTED'

# Call functions
>>> len(request.email_thread)
150

# Modify variables (for testing)
>>> request.ooo_toggle = True
```

### 4. Watch Expressions

Add expressions to the Watch panel to monitor them:
```python
len(email_thread)
result.classification
response.get_lm_usage()
```

### 5. Keyboard Shortcuts

- `F9` - Toggle breakpoint on current line
- `F5` - Start/Continue debugging
- `F10` - Step Over
- `F11` - Step Into
- `Shift+F11` - Step Out
- `Shift+F5` - Stop debugging
- `Ctrl+Shift+F5` - Restart debugging

---

## 🎯 Common Debugging Scenarios

### Scenario 1: Debug a Classification Request

**Goal:** See exactly what happens when you classify an email.

**Steps:**
1. Set breakpoint at line 150 in `app/main.py` (start of `/api/classify`)
2. Start debugging (`F5`)
3. Send a request (use `./test_api.sh` or curl)
4. When it pauses:
   - Inspect `request.email_thread`
   - Step through with `F10` to see each step
   - Step into classifier with `F11` at line 128

### Scenario 2: Debug Rule-Based Overrides

**Goal:** See which override rules trigger.

**Steps:**
1. Open `app/classes/modules/classification/classifier_optimized.py`
2. Set breakpoints at:
   - Line 212: Where classification runs
   - Line 230: "will forward" override
   - Line 257: Data sourcing override
   - Line 385: Thread complete override
3. Send a test request
4. Watch which breakpoints hit

### Scenario 3: Debug a Failing Test

**Goal:** Understand why a test fails.

**Steps:**
1. Open `tests/test_classifier.py`
2. Find the failing test function
3. Set a breakpoint inside the test
4. Right-click the test → **Debug Test**
5. Inspect variables when it pauses
6. Step through to find the issue

### Scenario 4: Debug Token Usage Tracking

**Goal:** Verify token usage is calculated correctly.

**Steps:**
1. Set breakpoint at line 658 in `classifier_optimized.py` (where usage is captured)
2. Add Watch expressions:
   ```python
   combined_usage
   total_tokens
   self._last_usage
   ```
3. Run a classification
4. Inspect the usage data when it pauses

---

## 🐛 Troubleshooting VSCode Debug Issues

### Issue: "No module named 'app'"

**Fix:** Make sure you're using the correct Python interpreter:
1. Press `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Choose the Poetry virtual environment (should show path with `.venv`)

### Issue: Breakpoints are gray (unverified)

**Cause:** Code hasn't loaded yet or file path mismatch.

**Fix:**
- For server debugging: Breakpoints work after server starts and receives first request
- For tests: Make sure you're debugging the right file

### Issue: Environment variables not loading

**Fix:** Check `.env` file exists:
```bash
# In terminal
ls -la .env
```

If missing:
```bash
cp .env.example .env
# Edit .env and add your Azure credentials
```

### Issue: Can't install debugpy

**Fix:**
```bash
poetry add --group dev debugpy
```

---

## 📋 Quick Reference

### Start Server in Debug Mode
1. `Ctrl+Shift+D` - Open Debug Panel
2. Select "FastAPI: Debug Server"
3. `F5` - Start

### Debug Current Python File
1. Open the file
2. `Ctrl+Shift+D`
3. Select "Python: Current File"
4. `F5`

### Debug Tests
1. `Ctrl+Shift+D`
2. Select "Python: Run Tests"
3. `F5`

### Set Breakpoint
- Click in left margin (gutter)
- OR press `F9` on the line

### Step Through Code
- `F10` - Next line (don't enter functions)
- `F11` - Step into function
- `Shift+F11` - Step out of function
- `F5` - Continue to next breakpoint

---

## 🎓 Learning Path

### Beginner
1. Set a breakpoint in `app/main.py` line 150
2. Start debugging
3. Send a test request
4. Inspect the `request` variable
5. Step through with `F10`

### Intermediate
1. Set multiple breakpoints in different files
2. Use Watch expressions
3. Try conditional breakpoints
4. Use the Debug Console to inspect/modify variables

### Advanced
1. Debug complex flows (classifier → overrides → response)
2. Use logpoints for non-intrusive logging
3. Debug tests with breakpoints in both test and application code
4. Use "justMyCode": false to step into library code (DSPy, FastAPI)

---

## 💡 Pro Tips

1. **Use the Debug Console** - It's like an interactive Python shell at the breakpoint!

2. **Copy variable values** - Right-click a variable → "Copy Value" to save complex data structures

3. **Restart is faster than Stop+Start** - Use `Ctrl+Shift+F5` to restart the server quickly

4. **Debug multiple things** - You can run multiple debug sessions (server + tests)

5. **Hot reload works in debug mode** - Edit code and uvicorn will auto-reload (but breakpoints might not work until restart)

6. **Check the OUTPUT panel** - If something's wrong, check Output → Python for errors

---

## 📚 Additional Resources

- [VSCode Python Debugging](https://code.visualstudio.com/docs/python/debugging)
- [Debugging FastAPI in VSCode](https://fastapi.tiangolo.com/tutorial/debugging/)
- [VSCode Debugger Keyboard Shortcuts](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)

---

**Happy Debugging! 🐛🔍**

Press `F5` to get started!
