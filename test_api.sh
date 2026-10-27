#!/bin/bash

# Simple script to test the API endpoints

echo "=== Testing Email Classifier API ==="
echo ""

# Test 1: Health Check
echo "1. Testing /health endpoint..."
curl -s http://localhost:8000/health | jq .
echo ""

# Test 2: Basic Classification (INTERESTED)
echo "2. Testing classification - INTERESTED example..."
curl -s -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "email_thread": "Hi, I would love to learn more about your product. Can we schedule a call?",
    "thread_id": "test-001",
    "user_id": 1,
    "persona_id": 1
  }' | jq .
echo ""

# Test 3: NOT INTERESTED
echo "3. Testing classification - NOT INTERESTED example..."
curl -s -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "email_thread": "Thanks but we are not interested.",
    "thread_id": "test-002"
  }' | jq .
echo ""

# Test 4: DO NOT CONTACT
echo "4. Testing classification - DO NOT CONTACT example..."
curl -s -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "email_thread": "Please unsubscribe me from your emails.",
    "thread_id": "test-003"
  }' | jq .
echo ""

# Test 5: Invalid API Key
echo "5. Testing invalid API key (should fail)..."
curl -s -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: wrong-key" \
  -d '{
    "email_thread": "Test email",
    "thread_id": "test-004"
  }' | jq .
echo ""

echo "=== Tests Complete ==="
