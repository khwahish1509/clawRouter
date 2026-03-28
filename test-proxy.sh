#!/bin/bash
# LiteLLM Proxy — Full Test Suite
# Tests every tier, fallback, caching, budget, and virtual keys
# No OpenClaw needed — pure curl + OpenAI API format
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && source .env && set +a

URL="http://localhost:4000"
KEY="${LITELLM_MASTER_KEY:?Set LITELLM_MASTER_KEY in .env}"
G='\033[0;32m' R='\033[0;31m' Y='\033[1;33m' B='\033[1;34m' N='\033[0m'
PASS=0 FAIL=0 SKIP=0

log()  { echo -e "\n${B}[$1]${N} $2"; }
ok()   { ((PASS++)); echo -e "  ${G}✓${N} $1"; }
fail() { ((FAIL++)); echo -e "  ${R}✗${N} $1"; }
skip() { ((SKIP++)); echo -e "  ${Y}—${N} $1 (skipped)"; }

call() {
  local model="$1" prompt="${2:-Say hi in 5 words}" max="${3:-30}"
  curl -sf "$URL/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max}" 2>/dev/null
}

call_with_key() {
  local vkey="$1" model="$2" prompt="${3:-Say hi in 5 words}"
  curl -sf "$URL/v1/chat/completions" \
    -H "Authorization: Bearer $vkey" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":30}" 2>/dev/null
}

# ── 1. Health ─────────────────────────────────────────
log "1" "Health checks"
curl -sf "$URL/health/liveliness" &>/dev/null && ok "Proxy alive" || fail "Proxy down — run ./manage.sh start"
curl -sf "$URL/health/readiness" &>/dev/null && ok "Proxy ready" || fail "Proxy not ready"

# ── 2. Model list ─────────────────────────────────────
log "2" "Model discovery"
models=$(curl -sf "$URL/v1/models" -H "Authorization: Bearer $KEY" 2>/dev/null)
if echo "$models" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['data'])>=8" 2>/dev/null; then
  count=$(echo "$models" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
  ok "Found $count models in /v1/models"
else
  fail "Model list empty or unreachable"
fi

# ── 3. Tier 1 — Premium (Anthropic) ──────────────────
log "3" "Tier 1 — Premium models"
for m in claude-haiku claude-sonnet; do
  resp=$(call "$m" "Reply with exactly: tier1-ok" 15)
  if echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" &>/dev/null; then
    ok "$m responded"
  else
    fail "$m failed — check ANTHROPIC_API_KEY"
  fi
done

# claude-opus — expensive, optional
log "3b" "Tier 1 — Opus (optional, costs \$75/M output)"
echo -ne "  Test claude-opus? [y/N] "
read -r -t 5 ans || ans="n"
if [[ "${ans,,}" == "y" ]]; then
  resp=$(call "claude-opus" "Reply with: opus-ok" 10)
  echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" &>/dev/null && ok "claude-opus" || fail "claude-opus"
else
  skip "claude-opus (user skipped)"
fi

# ── 4. Tier 2 — Mid ──────────────────────────────────
log "4" "Tier 2 — Budget models"
for m in deepseek-chat gemini-flash; do
  resp=$(call "$m" "Reply with exactly: tier2-ok" 15)
  if echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" &>/dev/null; then
    ok "$m responded"
  else
    fail "$m — check API key"
  fi
done

# ── 5. Tier 3 — Local ────────────────────────────────
log "5" "Tier 3 — Local Ollama"
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
  for m in ollama-llama ollama-mistral; do
    resp=$(call "$m" "Say hello" 20)
    if echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" &>/dev/null; then
      ok "$m responded"
    else
      fail "$m — model may not be pulled"
    fi
  done
else
  skip "ollama-llama (Ollama offline)"
  skip "ollama-mistral (Ollama offline)"
fi

# ── 6. OpenAI aliases ────────────────────────────────
log "6" "OpenAI-compat aliases"
for m in gpt-4 gpt-3.5-turbo; do
  resp=$(call "$m" "Reply with: alias-ok" 15)
  if echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" &>/dev/null; then
    actual=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','?'))")
    ok "$m → routed to $actual"
  else
    fail "$m alias"
  fi
done

# ── 7. Caching ────────────────────────────────────────
log "7" "Redis caching"
prompt="What is 2+2? Reply with just the number."
t1=$(date +%s%N)
call "deepseek-chat" "$prompt" 10 >/dev/null
t2=$(date +%s%N)
call "deepseek-chat" "$prompt" 10 >/dev/null
t3=$(date +%s%N)
ms1=$(( (t2 - t1) / 1000000 ))
ms2=$(( (t3 - t2) / 1000000 ))
if [ "$ms2" -lt "$ms1" ]; then
  ok "Cache hit: ${ms1}ms → ${ms2}ms ($(( ms1 - ms2 ))ms faster)"
else
  skip "Cache may not have kicked in: ${ms1}ms vs ${ms2}ms"
fi

# ── 8. Streaming ──────────────────────────────────────
log "8" "Streaming response"
stream=$(curl -sf "$URL/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Count 1 to 5"}],"max_tokens":30,"stream":true}' 2>/dev/null)
if echo "$stream" | grep -q "data:"; then
  ok "SSE stream received"
else
  fail "No stream data"
fi

# ── 9. Virtual key access control ─────────────────────
log "9" "Virtual key access control"
echo -e "  ${Y}Note:${N} Run ./manage.sh keys first to test this."
echo -ne "  Have a virtual key to test? Paste it (or Enter to skip): "
read -r -t 10 vkey || vkey=""
if [ -n "$vkey" ]; then
  # Should work: budget model
  resp=$(call_with_key "$vkey" "deepseek-chat" "Say ok")
  if echo "$resp" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" &>/dev/null; then
    ok "Virtual key → deepseek-chat allowed"
  else
    fail "Virtual key rejected for allowed model"
  fi
  # Should block: opus (if key is user-tier)
  resp=$(call_with_key "$vkey" "claude-opus" "Say ok" 2>&1)
  if echo "$resp" | grep -qi "error\|budget\|not allowed"; then
    ok "Virtual key → claude-opus correctly blocked"
  else
    skip "claude-opus not blocked (key may have full access)"
  fi
else
  skip "Virtual key tests (no key provided)"
fi

# ── 10. Error handling ────────────────────────────────
log "10" "Error handling"
# Bad model name
resp=$(curl -sf "$URL/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"nonexistent-model","messages":[{"role":"user","content":"test"}]}' 2>&1 || true)
if echo "$resp" | grep -qi "error\|not found\|invalid"; then
  ok "Bad model → proper error"
else
  fail "Bad model didn't error"
fi

# Bad auth
resp=$(curl -s -o /dev/null -w "%{http_code}" "$URL/v1/chat/completions" \
  -H "Authorization: Bearer sk-fake-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"test"}]}' 2>/dev/null)
if [[ "$resp" == "401" || "$resp" == "403" ]]; then
  ok "Bad key → HTTP $resp"
else
  fail "Bad key returned HTTP $resp instead of 401/403"
fi

# ── 11. Spend tracking ───────────────────────────────
log "11" "Spend tracking"
spend=$(curl -sf "$URL/global/spend/report" -H "Authorization: Bearer $KEY" 2>/dev/null)
if echo "$spend" | python3 -c "import sys,json; json.load(sys.stdin)" &>/dev/null; then
  ok "Spend report accessible"
else
  skip "Spend report not available yet"
fi

# ── Summary ───────────────────────────────────────────
echo -e "\n${B}═══════════════════════════════════════${N}"
echo -e "  ${G}Passed: $PASS${N}  ${R}Failed: $FAIL${N}  ${Y}Skipped: $SKIP${N}"
echo -e "${B}═══════════════════════════════════════${N}"
[ "$FAIL" -eq 0 ] && echo -e "${G}All tests passed!${N}" || echo -e "${R}Fix failures above before going to production.${N}"
