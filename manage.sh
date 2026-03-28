#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && source .env && set +a

URL="http://localhost:4000"
G='\033[0;32m' R='\033[0;31m' Y='\033[1;33m' N='\033[0m'
ok()   { echo -e "  $1: ${G}OK${N}"; }
fail() { echo -e "  $1: ${R}FAIL${N}"; return 1; }
warn() { echo -e "  $1: ${Y}$2${N}"; }

preflight() {
  echo "Preflight:"
  docker info &>/dev/null && ok "Docker" || fail "Docker not running"
  [ -f .env ] && ok ".env" || fail ".env missing (cp .env.example .env)"
  for v in LITELLM_MASTER_KEY POSTGRES_PASSWORD REDIS_PASSWORD ANTHROPIC_API_KEY; do
    grep -qE "^${v}=.*(change-me|xxx)" .env 2>/dev/null && fail "$v is placeholder"
  done
  ok "Secrets"
  ! lsof -i :4000 &>/dev/null && ok "Port 4000" || fail "Port 4000 in use"
  curl -sf http://localhost:11434/api/tags &>/dev/null && ok "Ollama" || warn "Ollama" "offline — tier 3 unavailable"
}

cmd_start() {
  preflight
  echo -e "\nPulling images..."
  docker compose pull -q
  docker compose up -d
  echo -n "Waiting"
  for _ in $(seq 1 30); do
    curl -sf "$URL/health/liveliness" &>/dev/null && {
      echo -e "\n${G}Running${N} → $URL\nUI → $URL/ui"
      return
    }
    echo -n "." && sleep 2
  done
  echo -e "\n${R}Timeout${N} — docker compose logs litellm"
  return 1
}

cmd_stop()    { docker compose down; echo "Stopped."; }
cmd_restart() { docker compose down && cmd_start; }
cmd_logs()    { docker compose logs -f --tail=100 "${1:-litellm}"; }

cmd_health() {
  echo "Services:"
  for c in litellm_proxy litellm_db litellm_redis; do
    local s h
    s=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || echo "missing")
    h=$(docker inspect -f '{{.State.Health.Status}}' "$c" 2>/dev/null || echo "-")
    [[ "$s" == "running" && "$h" == "healthy" ]] && ok "$c" || echo -e "  $c: ${R}$s/$h${N}"
  done
  curl -sf "$URL/health/liveliness" &>/dev/null && ok "API" || echo -e "  API: ${R}down${N}"
  curl -sf http://localhost:11434/api/tags &>/dev/null && ok "Ollama" || warn "Ollama" "offline"
  echo -e "\nResources:"
  docker stats --no-stream --format "  {{.Name}}: {{.CPUPerc}} cpu / {{.MemUsage}}" \
    litellm_proxy litellm_db litellm_redis 2>/dev/null || echo "  unavailable"
}

cmd_keys() {
  local keyfile="${1:-keys.json}"
  [ -f "$keyfile" ] || { echo "Key file not found: $keyfile"; exit 1; }
  curl -sf "$URL/health/liveliness" &>/dev/null || { fail "Proxy not running"; }
  local KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY not set}"

  echo "Creating keys from $keyfile:"
  python3 -c "
import json, subprocess, sys
with open('$keyfile') as f:
    keys = json.load(f)
for k in keys:
    payload = json.dumps({
        'key_alias': k['alias'],
        'max_budget': k['budget'],
        'budget_duration': '30d',
        'models': k['models'],
        'max_parallel_requests': 10,
        'rpm_limit': k['rpm'],
        'tpm_limit': 100000
    })
    r = subprocess.run([
        'curl', '-sf', '-X', 'POST', '$URL/key/generate',
        '-H', 'Authorization: Bearer $KEY',
        '-H', 'Content-Type: application/json',
        '-d', payload
    ], capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
        token = data.get('key', 'ERROR')
        print(f\"  \033[32m{k['alias']}\033[0m: {token} (\${k['budget']}/mo, {k['rpm']} rpm)\")
    except:
        print(f\"  \033[31m{k['alias']}: FAILED\033[0m — {r.stdout or r.stderr}\")
  "
}

cmd_test() {
  local model="${1:-claude-haiku}"
  local KEY="${LITELLM_MASTER_KEY:?}"
  echo "Testing $model..."
  curl -s "$URL/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: LiteLLM proxy is working.\"}],\"max_tokens\":20}" \
    | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']['content']
    model = d.get('model','?')
    usage = d.get('usage',{})
    print(f'  Response: {msg}')
    print(f'  Model: {model}')
    print(f'  Tokens: {usage.get(\"prompt_tokens\",\"?\")}/{usage.get(\"completion_tokens\",\"?\")}')
except Exception as e:
    print(f'  Error: {e}')
    json.dump(d if 'd' in dir() else {}, sys.stdout, indent=2)
"
}

cmd_spend() {
  local KEY="${LITELLM_MASTER_KEY:?}"
  curl -sf "$URL/global/spend/report" \
    -H "Authorization: Bearer $KEY" \
    | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(json.dumps(d, indent=2))
except:
    print('Could not fetch spend report')
"
}

cmd_route_test() {
  echo "Testing router classification..."
  curl -sf "http://localhost:4001/router/test" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Accuracy: {d['accuracy']}\n\")
for r in d['results']:
    match = '\033[32m✓\033[0m' if r['match'] else '\033[31m✗\033[0m'
    print(f\"  {match} {r['prompt'][:55]:55s} → {r['classified_as']:15s} → {r['model_picked']} ({r['savings']})\")
"
}

cmd_route() {
  local prompt="${1:?Usage: $0 route \"your prompt here\"}"
  local preset="${2:-auto}"
  local KEY="${LITELLM_MASTER_KEY:?}"
  echo "Routing: \"${prompt:0:60}...\" (preset: $preset)"
  curl -s "http://localhost:4001/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$preset\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json; print(json.dumps('$prompt'))")}],\"max_tokens\":100}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('_router', {})
if r.get('routed'):
    print(f'  Task:     {r[\"task_type\"]}')
    print(f'  Model:    {r[\"selected\"]}')
    print(f'  Score:    {r[\"score\"]}')
    print(f'  Runner-up:{r[\"runner_up\"]}')
    print(f'  Savings:  {r[\"savings\"]}')
    print(f'  Latency:  {r.get(\"latency_ms\",\"?\")}ms')
msg = d.get('choices',[{}])[0].get('message',{}).get('content','?')
print(f'  Response: {msg[:100]}')
"
}

case "${1:-}" in
  start)      cmd_start ;;
  stop)       cmd_stop ;;
  restart)    cmd_restart ;;
  health)     cmd_health ;;
  keys)       cmd_keys "${2:-keys.json}" ;;
  logs)       cmd_logs "${2:-}" ;;
  test)       cmd_test "${2:-claude-haiku}" ;;
  spend)      cmd_spend ;;
  route-test) cmd_route_test ;;
  route)      cmd_route "${2:-Hello}" "${3:-auto}" ;;
  *)          echo "Usage: $0 {start|stop|restart|health|keys|logs|test|spend|route-test|route \"prompt\" [auto|auto-cheap|auto-quality|auto-fast]}" ;;
esac
