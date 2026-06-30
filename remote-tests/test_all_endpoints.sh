#!/bin/bash
#
# Remote endpoint tests for omnivoice-server deployed on K8s.
# All requests go through the inference proxy with Bearer token auth.
# SSL verification is disabled (self-signed cert / internal CA).
#
# Usage: ./test_all_endpoints.sh
#

set -euo pipefail

# --- Configuration ---
BASE_URL="https://inference.adeoaiengine.ecouncil.ae/models/b55d006f-49c2-494d-9b8e-7575741c0e70/proxy"
API_KEY="sk-ah3DuYR8o0DvA8hhIsowmx4-YiWHImksQbb94ArFFJM"
AUTH_HEADER="Authorization: Bearer ${API_KEY}"
OUTPUT_DIR="/tmp/remote-tts-test"

# Reference audio for clone tests (local QA sample)
REF_AUDIO="/home/jyao/ADEO/tts/omnivoice/omnivoice-server/samples/qa/A01_default_no_params.wav"

# curl common flags: disable SSL verification, follow redirects, 60s timeout
CURL_OPTS="-sk --max-time 120 --connect-timeout 10"

# Track results
PASS=0
FAIL=0
FAILED_TESTS=()

mkdir -p "${OUTPUT_DIR}"

# --- Helpers ---
run_test() {
    local name="$1"
    echo ""
    echo "=== ${name} ==="
}

check_http() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if [ "${actual}" = "${expected}" ]; then
        echo "  ✅ HTTP ${actual}"
        PASS=$((PASS + 1))
    else
        echo "  ❌ HTTP ${actual} (expected ${expected})"
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("${name}")
    fi
}

# ============================================================
# 1. GET /health
# ============================================================
run_test "1. GET /health"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" -H "${AUTH_HEADER}" \
    "${BASE_URL}/health")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}"
check_http "health" "200" "${HTTP_CODE}"

# ============================================================
# 2. GET /health-check
# ============================================================
run_test "2. GET /health-check"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" -H "${AUTH_HEADER}" \
    "${BASE_URL}/health-check")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}"
check_http "health-check" "200" "${HTTP_CODE}"

# ============================================================
# 3. GET /v1/models
# ============================================================
run_test "3. GET /v1/models"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" -H "${AUTH_HEADER}" \
    "${BASE_URL}/v1/models")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}" | python3 -m json.tool 2>/dev/null || echo "  ${BODY}"
check_http "models" "200" "${HTTP_CODE}"

# ============================================================
# 4. GET /v1/voices
# ============================================================
run_test "4. GET /v1/voices"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" -H "${AUTH_HEADER}" \
    "${BASE_URL}/v1/voices")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  voices: {d[\"total\"]} total')" 2>/dev/null || echo "  ${BODY}"
check_http "voices" "200" "${HTTP_CODE}"

# ============================================================
# 5. POST /v1/audio/speech (default voice)
# ============================================================
run_test "5. POST /v1/audio/speech (default)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/speech_default.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "omnivoice",
        "input": "Hello, this is a test of the OmniVoice text-to-speech system.",
        "response_format": "wav"
    }')
check_http "speech-default" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/speech_default.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/speech_default.wav" 2>/dev/null) bytes"

# ============================================================
# 6. POST /v1/audio/speech (design voice)
# ============================================================
run_test "6. POST /v1/audio/speech (design voice)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/speech_design.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "omnivoice",
        "input": "This voice has been designed with specific attributes.",
        "instructions": "female,british accent,young adult,high pitch",
        "response_format": "wav"
    }')
check_http "speech-design" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/speech_design.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/speech_design.wav" 2>/dev/null) bytes"

# ============================================================
# 7. POST /v1/audio/speech (preset: nova)
# ============================================================
run_test "7. POST /v1/audio/speech (preset nova)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/speech_nova.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "omnivoice",
        "input": "Testing the nova preset voice.",
        "voice": "nova",
        "response_format": "wav"
    }')
check_http "speech-nova" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/speech_nova.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/speech_nova.wav" 2>/dev/null) bytes"

# ============================================================
# 8. POST /v1/audio/speech (streaming PCM)
# ============================================================
run_test "8. POST /v1/audio/speech (streaming PCM)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/speech_stream.pcm" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "omnivoice",
        "input": "This is a longer text that will be streamed in chunks. Each sentence is synthesized and sent as soon as it is ready.",
        "stream": true,
        "response_format": "pcm",
        "position_temperature": 0.0
    }')
check_http "speech-stream" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/speech_stream.pcm" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/speech_stream.pcm" 2>/dev/null) bytes"

# ============================================================
# 9. POST /v1/audio/speech/clone (one-shot, WITH ref_text)
# ============================================================
run_test "9. POST /v1/audio/speech/clone (one-shot, with ref_text)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/clone_oneshot.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech/clone" \
    -H "${AUTH_HEADER}" \
    -F "text=This is one-shot voice cloning without saving a profile." \
    -F "ref_audio=@${REF_AUDIO}" \
    -F "ref_text=Hello, this is a test of the OmniVoice text-to-speech system." \
    -F "speed=1.0")
check_http "clone-oneshot" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/clone_oneshot.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/clone_oneshot.wav" 2>/dev/null) bytes"

# ============================================================
# 10. POST /v1/audio/speech/clone (one-shot, WITHOUT ref_text → Whisper ASR)
# ============================================================
run_test "10. POST /v1/audio/speech/clone (no ref_text → Whisper ASR)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/clone_no_reftext.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech/clone" \
    -H "${AUTH_HEADER}" \
    -F "text=This is voice cloning without providing reference text, triggering Whisper ASR." \
    -F "ref_audio=@${REF_AUDIO}" \
    -F "speed=1.0")
check_http "clone-no-reftext" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/clone_no_reftext.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/clone_no_reftext.wav" 2>/dev/null) bytes"

# ============================================================
# 11. POST /v1/voices/profiles (create)
# ============================================================
run_test "11. POST /v1/voices/profiles (create)"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" \
    -X POST "${BASE_URL}/v1/voices/profiles" \
    -H "${AUTH_HEADER}" \
    -F "profile_id=my_voice" \
    -F "ref_audio=@${REF_AUDIO}" \
    -F "ref_text=Hello, this is a test of the OmniVoice text-to-speech system." \
    -F "overwrite=true")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}" | python3 -m json.tool 2>/dev/null || echo "  ${BODY}"
check_http "profile-create" "201" "${HTTP_CODE}"

# ============================================================
# 12. GET /v1/voices/profiles/my_voice
# ============================================================
run_test "12. GET /v1/voices/profiles/my_voice"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" \
    -H "${AUTH_HEADER}" \
    "${BASE_URL}/v1/voices/profiles/my_voice")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}" | python3 -m json.tool 2>/dev/null || echo "  ${BODY}"
check_http "profile-get" "200" "${HTTP_CODE}"

# ============================================================
# 13. PATCH /v1/voices/profiles/my_voice (update ref_text)
# ============================================================
run_test "13. PATCH /v1/voices/profiles/my_voice (update)"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" \
    -X PATCH "${BASE_URL}/v1/voices/profiles/my_voice" \
    -H "${AUTH_HEADER}" \
    -F "ref_text=Updated reference text for the voice profile.")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}" | python3 -m json.tool 2>/dev/null || echo "  ${BODY}"
check_http "profile-update" "200" "${HTTP_CODE}"

# ============================================================
# 14. POST /v1/audio/speech (clone:my_voice profile)
# ============================================================
run_test "14. POST /v1/audio/speech (clone:profile)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/clone_profile.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/speech" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "omnivoice",
        "input": "This uses a stored clone profile for voice cloning.",
        "voice": "clone:my_voice",
        "response_format": "wav"
    }')
check_http "clone-profile" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/clone_profile.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/clone_profile.wav" 2>/dev/null) bytes"

# ============================================================
# 15. DELETE /v1/voices/profiles/my_voice
# ============================================================
run_test "15. DELETE /v1/voices/profiles/my_voice"
HTTP_CODE=$(curl ${CURL_OPTS} -o /dev/null -w "%{http_code}" \
    -X DELETE "${BASE_URL}/v1/voices/profiles/my_voice" \
    -H "${AUTH_HEADER}")
check_http "profile-delete" "204" "${HTTP_CODE}"

# ============================================================
# 16. GET /v1/voices/profiles/my_voice (after delete → 404)
# ============================================================
run_test "16. GET /v1/voices/profiles/my_voice (after delete)"
RESP=$(curl ${CURL_OPTS} -w "\n%{http_code}" \
    -H "${AUTH_HEADER}" \
    "${BASE_URL}/v1/voices/profiles/my_voice")
HTTP_CODE=$(echo "${RESP}" | tail -1)
BODY=$(echo "${RESP}" | sed '$d')
echo "  ${BODY}"
check_http "profile-after-delete" "404" "${HTTP_CODE}"

# ============================================================
# 17. POST /v1/audio/script (single_track)
# ============================================================
run_test "17. POST /v1/audio/script (single_track)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/script_single.wav" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/script" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "script": [
            {"speaker": "narrator", "text": "Welcome to the OmniVoice script synthesis demo.", "voice": "openai:onyx"},
            {"speaker": "alice", "text": "Hello! I am Alice, and I will be your guide today.", "voice": "openai:nova"},
            {"speaker": "narrator", "text": "Thank you Alice. Let us begin.", "voice": "openai:onyx"}
        ],
        "default_voice": "openai:onyx",
        "speed": 1.0,
        "on_error": "abort",
        "pause_between_speakers": 0.3,
        "response_format": "wav",
        "output_format": "single_track"
    }')
check_http "script-single" "200" "${HTTP_CODE}"
file "${OUTPUT_DIR}/script_single.wav" 2>/dev/null && echo "  size: $(stat -c%s "${OUTPUT_DIR}/script_single.wav" 2>/dev/null) bytes"

# ============================================================
# 18. POST /v1/audio/script (multi_track)
# ============================================================
run_test "18. POST /v1/audio/script (multi_track)"
HTTP_CODE=$(curl ${CURL_OPTS} -o "${OUTPUT_DIR}/script_multi.json" \
    -w "%{http_code}" \
    -X POST "${BASE_URL}/v1/audio/script" \
    -H "${AUTH_HEADER}" \
    -H "Content-Type: application/json" \
    -d '{
        "script": [
            {"speaker": "narrator", "text": "Welcome to the OmniVoice script synthesis demo.", "voice": "openai:onyx"},
            {"speaker": "alice", "text": "Hello! I am Alice, and I will be your guide today.", "voice": "openai:nova"}
        ],
        "default_voice": "openai:onyx",
        "speed": 1.0,
        "on_error": "abort",
        "pause_between_speakers": 0.3,
        "response_format": "wav",
        "output_format": "multi_track"
    }')
check_http "script-multi" "200" "${HTTP_CODE}"
echo "  Speakers:" $(python3 -c "import json; d=json.load(open('${OUTPUT_DIR}/script_multi.json')); print(list(d.get('tracks',{}).keys()))" 2>/dev/null || echo "(parse failed)")

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "  RESULTS: ${PASS} passed, ${FAIL} failed"
echo "============================================"
if [ ${FAIL} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  ❌ ${t}"
    done
    exit 1
fi
echo "All tests passed! ✅"
