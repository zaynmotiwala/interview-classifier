#!/bin/bash
# Setup script to fix VSCode debugging

echo "🔧 Setting up VSCode debugging for interview project..."
echo ""

# Step 1: Install debugpy if not present
echo "1. Checking debugpy..."
poetry run python -c "import debugpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✅ debugpy already installed"
else
    echo "   ⚠️  Installing debugpy..."
    poetry run pip install debugpy
fi

echo ""

# Step 2: Verify Python interpreter
echo "2. Python interpreter:"
PYTHON_PATH=$(poetry env info --path)
echo "   📍 $PYTHON_PATH/Scripts/python.exe"

echo ""

# Step 3: Test imports
echo "3. Testing critical imports..."
poetry run python -c "import fastapi; print('   ✅ fastapi')" 2>&1
poetry run python -c "import uvicorn; print('   ✅ uvicorn')" 2>&1
poetry run python -c "import pytest; print('   ✅ pytest')" 2>&1

echo ""

# Step 4: Check .env file
echo "4. Checking .env file..."
if [ -f .env ]; then
    echo "   ✅ .env exists"
    # Check if Azure keys are set
    if grep -q "AZURE_API_KEY=your_azure" .env; then
        echo "   ⚠️  WARNING: .env still has placeholder values"
        echo "   → Edit .env and add your Azure OpenAI credentials"
    else
        echo "   ✅ .env appears configured"
    fi
else
    echo "   ❌ .env missing!"
    echo "   → Run: cp .env.example .env"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps in VSCode:"
echo "   1. Press Ctrl+Shift+P"
echo "   2. Type: 'Python: Select Interpreter'"
echo "   3. Choose: revreply-prompt-engine-2QJ1TntQ-py3.12"
echo "   4. Press F5 to start debugging!"
echo ""
