#!/bin/bash
# End-to-end interview API test script
# Usage: ./test-flow.sh [base_url]
# Requires: curl, jq

BASE_URL="${1:-http://localhost:3000}"
COOKIE="" # Set auth cookie here or pass via environment

echo "=== Interview API E2E Test ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Unauthenticated request should fail
echo "[1/6] Testing auth guard..."
UNAUTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/interview/start" \
  -X POST -H "Content-Type: application/json" \
  -d '{"topic":"arrays","difficulty":"medium","language":"javascript"}')
if [ "$UNAUTH" = "401" ]; then
  echo "  PASS: Unauthenticated request returned 401"
else
  echo "  FAIL: Expected 401, got $UNAUTH"
fi

# Test 2: Start interview
echo "[2/6] Starting interview..."
START_RESPONSE=$(curl -s "$BASE_URL/api/interview/start" \
  -X POST -H "Content-Type: application/json" \
  -H "Cookie: $COOKIE" \
  -d '{"topic":"arrays","difficulty":"medium","language":"javascript"}')
INTERVIEW_ID=$(echo "$START_RESPONSE" | jq -r '.id // empty')
if [ -n "$INTERVIEW_ID" ]; then
  echo "  PASS: Interview started with ID: $INTERVIEW_ID"
  QUESTION=$(echo "$START_RESPONSE" | jq -r '.question_title // empty')
  echo "  Question: $QUESTION"
else
  echo "  FAIL: No interview ID returned"
  echo "  Response: $START_RESPONSE"
  exit 1
fi

# Test 3: Send candidate message (approach discussion)
echo "[3/6] Sending candidate message (approach)..."
MSG1_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/interview/message" \
  -X POST -H "Content-Type: application/json" \
  -H "Cookie: $COOKIE" \
  -d "{\"interview_id\":\"$INTERVIEW_ID\",\"message\":\"I think I can use a hash map approach for O(n) time complexity.\",\"code_snapshot\":\"\"}")
if [ "$MSG1_STATUS" = "200" ]; then
  echo "  PASS: Message sent, streaming response received"
else
  echo "  FAIL: Expected 200, got $MSG1_STATUS"
fi

# Test 4: Send candidate message with code
echo "[4/6] Sending candidate message (with code)..."
MSG2_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/interview/message" \
  -X POST -H "Content-Type: application/json" \
  -H "Cookie: $COOKIE" \
  -d "{\"interview_id\":\"$INTERVIEW_ID\",\"message\":\"Here is my solution.\",\"code_snapshot\":\"function twoSum(nums, target) {\\n  const map = new Map();\\n  for (let i = 0; i < nums.length; i++) {\\n    const complement = target - nums[i];\\n    if (map.has(complement)) return [map.get(complement), i];\\n    map.set(nums[i], i);\\n  }\\n}\"}")
if [ "$MSG2_STATUS" = "200" ]; then
  echo "  PASS: Code message sent, streaming response received"
else
  echo "  FAIL: Expected 200, got $MSG2_STATUS"
fi

# Test 5: End interview
echo "[5/6] Ending interview..."
END_RESPONSE=$(curl -s "$BASE_URL/api/interview/end" \
  -X POST -H "Content-Type: application/json" \
  -H "Cookie: $COOKIE" \
  -d "{\"interview_id\":\"$INTERVIEW_ID\"}")
OVERALL_SCORE=$(echo "$END_RESPONSE" | jq -r '.overall_score // empty')
if [ -n "$OVERALL_SCORE" ] && [ "$OVERALL_SCORE" -ge 1 ] && [ "$OVERALL_SCORE" -le 10 ]; then
  echo "  PASS: Scorecard received — overall score: $OVERALL_SCORE/10"
  echo "  Problem solving: $(echo "$END_RESPONSE" | jq -r '.problem_solving_score')/10"
  echo "  Code quality: $(echo "$END_RESPONSE" | jq -r '.code_quality_score')/10"
  echo "  Communication: $(echo "$END_RESPONSE" | jq -r '.communication_score')/10"
else
  echo "  FAIL: Invalid or missing scorecard"
  echo "  Response: $END_RESPONSE"
fi

# Test 6: Ending already-ended interview should fail
echo "[6/6] Testing double-end guard..."
DOUBLE_END=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/interview/end" \
  -X POST -H "Content-Type: application/json" \
  -H "Cookie: $COOKIE" \
  -d "{\"interview_id\":\"$INTERVIEW_ID\"}")
if [ "$DOUBLE_END" = "400" ]; then
  echo "  PASS: Double-end returned 400"
else
  echo "  FAIL: Expected 400, got $DOUBLE_END"
fi

echo ""
echo "=== Tests complete ==="
