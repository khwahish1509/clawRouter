"""
clawRouter — smart LLM model selector for OpenClaw.
Replaces OpenRouter. Self-hosted. Learns from your usage. Zero markup.

Send model="auto" → analyzes your prompt → picks cheapest model that works.
Every request is logged. Quality scores update from real data. Gets smarter over time.

Client (port 4001) → clawRouter → LiteLLM (port 4000) → Providers

Presets: auto | auto-quality | auto-cheap | auto-fast
"""

import re, os, json, time, logging, yaml, hashlib, threading
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
import httpx

LITELLM = os.getenv("LITELLM_URL", "http://litellm:4000")
CFG_PATH = os.getenv("ROUTER_CONFIG", "/app/router_config.yaml")
LOG_DIR = Path(os.getenv("ROUTER_LOG_DIR", "/app/logs"))
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [router] %(message)s")
log = logging.getLogger("router")

# ═══════════════════════════════════════════════════════════════════
# 1. MODEL REGISTRY — costs, context, speed, baseline quality
#    Quality scores are STARTING DEFAULTS. The learning engine
#    overrides these with real data after ~100 requests per task.
# ═══════════════════════════════════════════════════════════════════
MODELS = {
    "claude-opus":    {"ci":15.0,  "co":75.0,  "ctx":200000, "spd":0.30, "q":{"code":0.97,"review":0.98,"reason":0.99,"creative":0.95,"chat":0.90,"summary":0.92,"extract":0.90,"math":0.95,"translate":0.90}},
    "claude-sonnet":  {"ci":3.0,   "co":15.0,  "ctx":200000, "spd":0.50, "q":{"code":0.95,"review":0.93,"reason":0.95,"creative":0.92,"chat":0.88,"summary":0.90,"extract":0.88,"math":0.90,"translate":0.88}},
    "claude-haiku":   {"ci":0.80,  "co":4.0,   "ctx":200000, "spd":0.85, "q":{"code":0.80,"review":0.75,"reason":0.75,"creative":0.80,"chat":0.85,"summary":0.85,"extract":0.85,"math":0.75,"translate":0.82}},
    "deepseek-chat":  {"ci":0.14,  "co":0.28,  "ctx":64000,  "spd":0.75, "q":{"code":0.90,"review":0.85,"reason":0.80,"creative":0.75,"chat":0.85,"summary":0.82,"extract":0.80,"math":0.85,"translate":0.78}},
    "deepseek-coder": {"ci":0.14,  "co":0.28,  "ctx":64000,  "spd":0.75, "q":{"code":0.93,"review":0.90,"reason":0.70,"creative":0.55,"chat":0.65,"summary":0.60,"extract":0.75,"math":0.80,"translate":0.50}},
    "gemini-flash":   {"ci":0.075, "co":0.30,  "ctx":1000000,"spd":0.95, "q":{"code":0.75,"review":0.70,"reason":0.72,"creative":0.75,"chat":0.82,"summary":0.85,"extract":0.82,"math":0.78,"translate":0.80}},
    "ollama-llama":   {"ci":0.0,   "co":0.0,   "ctx":8192,   "spd":0.60, "q":{"code":0.55,"review":0.45,"reason":0.40,"creative":0.50,"chat":0.70,"summary":0.55,"extract":0.50,"math":0.35,"translate":0.45}},
    "ollama-mistral": {"ci":0.0,   "co":0.0,   "ctx":8192,   "spd":0.60, "q":{"code":0.50,"review":0.45,"reason":0.40,"creative":0.55,"chat":0.70,"summary":0.55,"extract":0.50,"math":0.35,"translate":0.50}},
}

# ═══════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION — structural analysis of the prompt
#    Not just regex keywords. Measures actual characteristics.
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Features:
    token_est: int          # estimated token count
    turns: int              # conversation length (multi-turn = harder)
    code_blocks: int        # ``` count (code presence)
    code_density: float     # % of text that looks like code (0-1)
    question_count: int     # number of ? marks
    lang_mentioned: str     # programming language if any
    has_error_trace: bool   # stack trace / error present
    avg_word_len: float     # technical text has longer words
    unique_ratio: float     # vocabulary richness (unique/total words)
    line_count: int         # many lines = structured content
    system_prompt_len: int  # system prompt token estimate

def extract_features(messages: list[dict]) -> Features:
    """Extract structural features from the conversation."""
    user_parts, sys_parts = [], []
    for m in messages:
        content = m.get("content", "")
        if not isinstance(content, str): continue
        if m.get("role") == "system": sys_parts.append(content)
        elif m.get("role") == "user": user_parts.append(content)

    text = "\n".join(user_parts)
    all_text = "\n".join(sys_parts + user_parts)
    words = re.findall(r"\w+", all_text.lower())

    # Code density: count chars inside code blocks vs total
    code_chars = sum(len(b) for b in re.findall(r"```[\s\S]*?```", text))
    code_density = code_chars / max(len(text), 1)

    # Programming language detection
    lang_pat = r"\b(python|javascript|typescript|java|rust|go|golang|c\+\+|cpp|ruby|php|swift|kotlin|scala|r|sql|bash|shell)\b"
    langs = re.findall(lang_pat, all_text.lower())

    return Features(
        token_est=max(len(all_text) // 4, 1),
        turns=sum(1 for m in messages if m.get("role") == "user"),
        code_blocks=all_text.count("```") // 2,
        code_density=round(code_density, 3),
        question_count=text.count("?"),
        lang_mentioned=langs[0] if langs else "",
        has_error_trace=bool(re.search(r"(Traceback|Error:|Exception|FATAL|panic:|at \w+\.java:\d+|\.py.*line \d+)", all_text)),
        avg_word_len=round(sum(len(w) for w in words) / max(len(words), 1), 2),
        unique_ratio=round(len(set(words)) / max(len(words), 1), 3),
        line_count=all_text.count("\n") + 1,
        system_prompt_len=max(len("\n".join(sys_parts)) // 4, 0),
    )

# ═══════════════════════════════════════════════════════════════════
# 3. CLASSIFIER — uses features + patterns for robust classification
# ═══════════════════════════════════════════════════════════════════
TASK_SIGNALS = {
    "code":      [r"```", r"\b(implement|build|create|write|code)\b.*\b(function|class|api|endpoint|component|module|script|app|program|algorithm|sort|server)\b",
                  r"\b(write|implement|create|build|code)\b.*\bin (python|javascript|typescript|java|rust|go|c\+\+|ruby|php)\b"],
    "review":    [r"\b(fix\w*|debug\w*|bug\w*|error\w*|traceback|exception|crash\w*|broken|fail\w*|TypeError|ValueError|undefined|null|wrong|issue)\b",
                  r"\b(refactor|optimize|improve|clean.?up|lint|migrate|upgrade)\b.*\b(code|function|class|module|codebase)\b",
                  r"\b(review|pr|pull.request|diff|commit|merge)\b"],
    "reason":    [r"\b(explain|why|how.does|analyze|compare|evaluate|pros?.and?.cons|trade.?off|difference.between)\b",
                  r"\b(step.by.step|think.through|reason|logic|argument|critique|assess|design|architect)\b"],
    "math":      [r"\b(calculate|compute|formula|equation|statistic|probability|average|median|percent|sum|multiply|divide)\b",
                  r"\b(data.analy|chart|graph|plot|regression|dataset|csv|sql|query|aggregate|pivot)\b"],
    "creative":  [r"\b(write|draft|compose)\b.*\b(story|blog|essay|poem|article|post|email|letter|copy|script|song)\b",
                  r"\b(creative|imagine|brainstorm|narrative|fiction|copywriting|rewrite|tone)\b"],
    "summary":   [r"\b(summar\w*|tldr|tl;dr|condense|key.?points|digest|recap|gist|brief|shorten|overview)\b"],
    "extract":   [r"\b(extract|parse|find.all|list.all|pull.out|scrape|identify|detect|categorize|classify|tag|filter)\b"],
    "translate":  [r"\b(translate|translation|translat)\b",
                  r"\bin (spanish|french|german|chinese|japanese|hindi|arabic|korean|portuguese|russian|italian|dutch)\b"],
}

def classify(feat: Features, text: str) -> tuple[str, float]:
    """Classify task type and estimate complexity. Returns (task, complexity)."""
    text_lower = text.lower()

    # ── Pattern scoring ───────────────────────────────
    scores = {}
    for task, patterns in TASK_SIGNALS.items():
        score = 0.0
        for pat in patterns:
            hits = len(re.findall(pat, text_lower))
            score += hits * 2.0
        scores[task] = score

    # ── Feature-based boosts (the smart part) ─────────
    # Code features: code blocks, language mentions, code density
    if feat.code_blocks > 0:
        scores["code"] = scores.get("code", 0) + feat.code_blocks * 2.0
        # Code + error/fix language = code REVIEW, not generation
        if feat.has_error_trace or scores.get("review", 0) > 0:
            scores["review"] = scores.get("review", 0) + 8.0
            scores["code"] = max(scores.get("code", 0) - 3.0, 0)  # demote code-gen
    if feat.code_density > 0.3 and scores.get("review", 0) == 0:
        scores["code"] = scores.get("code", 0) + 3.0
    if feat.lang_mentioned and scores.get("review", 0) == 0:
        scores["code"] = scores.get("code", 0) + 2.0
    if feat.has_error_trace and feat.code_blocks == 0:
        scores["review"] = scores.get("review", 0) + 4.0

    # Multi-turn conversations with many questions = reasoning
    if feat.turns > 3:
        scores["reason"] = scores.get("reason", 0) + 2.0
    if feat.question_count >= 3:
        scores["reason"] = scores.get("reason", 0) + 1.5

    # Long technical text = probably needs summarization or reasoning
    if feat.avg_word_len > 5.5 and feat.token_est > 500:
        scores["reason"] = scores.get("reason", 0) + 1.0

    # ── Pick winner ───────────────────────────────────
    if max(scores.values(), default=0) > 0:
        task = max(scores, key=scores.get)
    else:
        task = "chat" if feat.token_est < 200 else "reason"

    # Only override to chat if genuinely no meaningful signal detected
    max_signal = max(scores.values(), default=0)
    if feat.token_est < 6 and max_signal == 0:
        task = "chat"

    # ── Complexity: 0-1 based on multiple signals ─────
    cx = 0.0
    cx += min(0.30, feat.token_est / 10000)                          # length factor
    cx += min(0.15, feat.turns * 0.03)                                # multi-turn
    cx += min(0.15, feat.question_count * 0.05)                       # questions
    cx += 0.10 if feat.code_blocks >= 2 else 0.0                     # multiple code blocks
    cx += 0.10 if feat.avg_word_len > 5.5 else 0.0                   # technical vocabulary
    cx += 0.10 if feat.system_prompt_len > 200 else 0.0              # heavy system prompt
    cx += 0.10 if bool(re.search(r"\b(step.by.step|detailed|comprehensive|thorough|production|enterprise)\b", text_lower)) else 0.0
    cx = round(min(1.0, cx), 3)

    return task, cx

# ═══════════════════════════════════════════════════════════════════
# 4. LEARNING ENGINE — logs requests, builds quality matrix from data
# ═══════════════════════════════════════════════════════════════════
_log_lock = threading.Lock()
_learned_scores: dict[str, dict[str, float]] = {}  # model → task → learned_quality
_log_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "retry_count": 0, "total_latency": 0, "total_tokens": 0}))

def _log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"requests_{time.strftime('%Y%m')}.jsonl"

def log_request(entry: dict):
    """Append request log entry (non-blocking)."""
    def _write():
        with _log_lock:
            with open(_log_path(), "a") as f:
                f.write(json.dumps(entry) + "\n")
            # Update in-memory stats
            model = entry.get("model", "")
            task = entry.get("task", "")
            if model and task:
                s = _log_stats[model][task]
                s["count"] += 1
                s["total_latency"] += entry.get("latency_ms", 0)
                s["total_tokens"] += entry.get("output_tokens", 0)
                if entry.get("is_retry"): s["retry_count"] += 1
    threading.Thread(target=_write, daemon=True).start()

def learn_from_logs():
    """Read historical logs, compute quality scores per model per task.
    Quality signal: if a model has low retry rate and good token output, it's good.
    Called on startup and periodically."""
    global _learned_scores
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "retries": 0, "avg_latency": 0}))

    for logfile in sorted(LOG_DIR.glob("requests_*.jsonl")):
        try:
            with open(logfile) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        model, task = e.get("model", ""), e.get("task", "")
                        if not model or not task: continue
                        s = stats[model][task]
                        s["count"] += 1
                        if e.get("is_retry"): s["retries"] += 1
                        s["avg_latency"] += e.get("latency_ms", 0)
                    except json.JSONDecodeError: continue
        except: continue

    learned = {}
    for model, tasks in stats.items():
        learned[model] = {}
        for task, s in tasks.items():
            if s["count"] < 20: continue  # need minimum data
            retry_rate = s["retries"] / s["count"]
            # Quality = 1.0 - retry_rate, clamped to [0.3, 1.0]
            # Retry is the strongest signal: if users retry, the model wasn't good enough
            learned[model][task] = round(max(0.3, min(1.0, 1.0 - retry_rate * 2)), 3)
    _learned_scores = learned
    if learned:
        log.info(f"Learned quality scores for {sum(len(t) for t in learned.values())} model-task pairs")

def get_quality(model: str, task: str) -> float:
    """Get quality score: learned data takes priority over defaults."""
    if model in _learned_scores and task in _learned_scores[model]:
        return _learned_scores[model][task]
    return MODELS.get(model, {}).get("q", {}).get(task, 0.50)

# ═══════════════════════════════════════════════════════════════════
# 5. SELECTOR — scores models using features + learned quality
# ═══════════════════════════════════════════════════════════════════
PRESETS = {
    "balanced": (0.35, 0.45, 0.20),
    "quality":  (0.70, 0.10, 0.20),
    "cheap":    (0.15, 0.70, 0.15),
    "fast":     (0.15, 0.15, 0.70),
}

@dataclass
class Route:
    task: str; complexity: float; model: str; score: float
    runner_up: str; savings: str; reason: str

def select(task: str, cx: float, feat: Features, cfg: dict, preset: str = "balanced") -> Route:
    """Score all models, return the best pick."""
    wq, wc, ws = PRESETS.get(preset, PRESETS["balanced"])

    # Dynamic quality floor based on complexity
    min_q = cfg.get("min_quality", 0.50)
    if cx > 0.7: min_q = max(min_q, 0.75)
    elif cx > 0.4: min_q = max(min_q, 0.60)

    # Overrides
    overrides = cfg.get("overrides", {})
    if task in overrides and overrides[task] and overrides[task] in MODELS:
        return Route(task, cx, overrides[task], 1.0, "-", "override", "config override")

    # Budget guard
    bg = cfg.get("budget_guard", {})
    allowed = set(bg["cheap_models"]) if bg.get("enabled") and bg.get("force_cheap") else None

    max_cost = max((m["ci"]+m["co"])/2 for m in MODELS.values()) or 1
    candidates = []

    for name, m in MODELS.items():
        if allowed and name not in allowed: continue
        q = get_quality(name, task)  # uses learned scores if available
        if q < min_q: continue
        if feat.token_est > int(m["ctx"] * 0.8): continue
        if feat.token_est > 6000 and m["ctx"] < 32000: continue

        cost_s = 1.0 - ((m["ci"]+m["co"])/2) / max_cost
        final = q * wq + cost_s * wc + m["spd"] * ws
        candidates.append((round(final, 4), name, f"q={q:.2f}|c={cost_s:.2f}|s={m['spd']:.2f}"))

    if not candidates:
        return Route(task, cx, "claude-haiku", 0.0, "-", "fallback", "no model met quality floor")

    candidates.sort(reverse=True)
    winner_score, winner_name, winner_reason = candidates[0]
    runner = candidates[1][1] if len(candidates) > 1 else "-"

    # Calculate savings
    opus_cost = (MODELS["claude-opus"]["ci"] + MODELS["claude-opus"]["co"]) / 2
    my_cost = (MODELS[winner_name]["ci"] + MODELS[winner_name]["co"]) / 2
    savings = f"{round((1 - my_cost/opus_cost)*100)}% cheaper" if my_cost < opus_cost else "premium"

    return Route(task, cx, winner_name, winner_score, runner, savings, winner_reason)

# ═══════════════════════════════════════════════════════════════════
# 6. RETRY DETECTION — tracks if user retries = model wasn't good
# ═══════════════════════════════════════════════════════════════════
_recent_requests: dict[str, dict] = {}  # hash → {model, task, time}

def _prompt_hash(messages: list[dict]) -> str:
    """Hash the user's prompt to detect retries."""
    text = "".join(m.get("content","") for m in messages if m.get("role") == "user")
    return hashlib.md5(text.encode()).hexdigest()[:12]

def detect_retry(messages: list[dict], current_model: str) -> bool:
    """If same prompt was sent recently with a different model, it's a retry."""
    h = _prompt_hash(messages)
    prev = _recent_requests.get(h)
    is_retry = prev is not None and prev["model"] != current_model and (time.time() - prev["time"]) < 300
    _recent_requests[h] = {"model": current_model, "task": "", "time": time.time()}
    # Cleanup old entries
    if len(_recent_requests) > 10000:
        cutoff = time.time() - 300
        for k in list(_recent_requests):
            if _recent_requests[k]["time"] < cutoff: del _recent_requests[k]
    return is_retry

# ═══════════════════════════════════════════════════════════════════
# 7. SERVER
# ═══════════════════════════════════════════════════════════════════
_cfg = {}
_client = httpx.AsyncClient(timeout=120.0)
_stats = {"total": 0, "routed": 0, "passthrough": 0, "total_saved_pct": 0}
_cost_tracker = {"total_actual": 0.0, "total_if_opus": 0.0}  # running cost comparison

@asynccontextmanager
async def lifespan(_):
    global _cfg
    try:
        with open(CFG_PATH) as f: _cfg.update(yaml.safe_load(f) or {})
    except FileNotFoundError: pass
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    learn_from_logs()
    log.info(f"clawRouter ready → {LITELLM}")
    yield

app = FastAPI(title="clawRouter", version="2.0.0", lifespan=lifespan)
AUTO = {"auto", "auto-quality", "auto-cheap", "auto-fast"}

@app.post("/v1/chat/completions")
async def route(req: Request):
    body = await req.json()
    auth = req.headers.get("authorization", "")
    model_req = body.get("model", "auto")
    messages = body.get("messages", [])
    _stats["total"] += 1

    if model_req in AUTO:
        _stats["routed"] += 1
        preset = model_req.split("-", 1)[1] if "-" in model_req else "balanced"
        feat = extract_features(messages)
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        task, cx = classify(feat, text)
        route_result = select(task, cx, feat, _cfg, preset)
        body["model"] = route_result.model
        is_retry = detect_retry(messages, route_result.model)

        meta = {
            "routed": True, "task": route_result.task, "complexity": route_result.complexity,
            "model": route_result.model, "score": route_result.score, "runner_up": route_result.runner_up,
            "savings": route_result.savings, "reason": route_result.reason, "preset": preset,
            "features": {"tokens": feat.token_est, "turns": feat.turns, "code_blocks": feat.code_blocks,
                         "code_density": feat.code_density, "questions": feat.question_count,
                         "lang": feat.lang_mentioned, "has_error": feat.has_error_trace},
        }
        log.info(f"{task}|cx={cx}|tok~{feat.token_est} → {route_result.model} ({route_result.savings})")
    else:
        _stats["passthrough"] += 1
        feat = extract_features(messages)
        task, cx = classify(feat, " ".join(m.get("content","") for m in messages if isinstance(m.get("content"),str)))
        is_retry = detect_retry(messages, model_req)
        meta = {"routed": False, "model": model_req}
        route_result = None

    headers = {"Authorization": auth, "Content-Type": "application/json"}
    try:
        if body.get("stream"):
            r = _client.build_request("POST", f"{LITELLM}/v1/chat/completions", json=body, headers=headers)
            resp = await _client.send(r, stream=True)
            async def stream():
                try:
                    async for chunk in resp.aiter_bytes(): yield chunk
                finally: await resp.aclose()
            return StreamingResponse(stream(), status_code=resp.status_code, media_type="text/event-stream",
                                     headers={"X-Router-Model": meta.get("model",""), "X-Router-Task": task if route_result else ""})
        else:
            t0 = time.monotonic()
            resp = await _client.post(f"{LITELLM}/v1/chat/completions", json=body, headers=headers)
            latency = round((time.monotonic() - t0) * 1000)
            data = resp.json()

            # Extract response metadata for logging
            usage = data.get("usage", {})
            out_tokens = usage.get("completion_tokens", 0)
            in_tokens = usage.get("prompt_tokens", 0)

            # Track costs
            chosen = MODELS.get(body["model"], {})
            actual_cost = (in_tokens * chosen.get("ci", 0) + out_tokens * chosen.get("co", 0)) / 1_000_000
            opus_cost = (in_tokens * MODELS["claude-opus"]["ci"] + out_tokens * MODELS["claude-opus"]["co"]) / 1_000_000
            _cost_tracker["total_actual"] += actual_cost
            _cost_tracker["total_if_opus"] += opus_cost

            # Log for learning
            log_request({
                "ts": time.time(), "task": task, "model": body["model"],
                "preset": meta.get("preset", "direct"), "complexity": cx,
                "in_tokens": in_tokens, "output_tokens": out_tokens,
                "latency_ms": latency, "is_retry": is_retry,
                "cost_actual": round(actual_cost, 6), "cost_opus": round(opus_cost, 6),
                "features": {"tokens": feat.token_est, "turns": feat.turns, "code_blocks": feat.code_blocks},
            })

            data["_router"] = {**meta, "ms": latency}
            return JSONResponse(data, status_code=resp.status_code,
                                headers={"X-Router-Model": body["model"], "X-Router-Task": task})
    except httpx.HTTPError as e:
        return JSONResponse({"error": str(e)}, status_code=502)

# ═══════════════════════════════════════════════════════════════════
# 8. DASHBOARD — shows cost savings, task distribution, model usage
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>clawRouter Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px}
h1{font-size:20px;color:#58a6ff;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px}
.card h2{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.big{font-size:36px;font-weight:700;color:#58a6ff}.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}
table{width:100%;border-collapse:collapse;margin-top:8px}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #21262d}
th{color:#8b949e;font-size:12px;text-transform:uppercase}td{font-size:14px}
.bar{height:8px;border-radius:4px;background:#21262d;overflow:hidden;margin-top:4px}
.bar-fill{height:100%;border-radius:4px;background:#58a6ff}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.tag-code{background:#1f3a5f;color:#58a6ff}.tag-chat{background:#1a3328;color:#3fb950}
.tag-reason{background:#3d2e00;color:#d29922}.tag-other{background:#2d1f3d;color:#bc8cff}
footer{text-align:center;color:#484f58;font-size:12px;margin-top:24px}
</style></head><body>
<h1>clawRouter Dashboard</h1>
<div class="grid">
  <div class="card"><h2>Total Requests</h2><div class="big">{total}</div>
    <div style="margin-top:8px;font-size:13px">{routed} routed · {passthrough} passthrough</div></div>
  <div class="card"><h2>Total Cost</h2><div class="big green">${actual:.4f}</div>
    <div style="margin-top:8px;font-size:13px">vs ${opus:.4f} if all Opus</div></div>
  <div class="card"><h2>Money Saved</h2><div class="big yellow">${saved:.4f}</div>
    <div style="margin-top:8px;font-size:13px">{pct:.1f}% reduction</div></div>
  <div class="card"><h2>Learning Status</h2><div class="big">{learned_pairs}</div>
    <div style="margin-top:8px;font-size:13px">model-task pairs learned from data</div></div>
</div>
<div class="grid">
  <div class="card"><h2>Model Usage</h2><table><tr><th>Model</th><th>Requests</th><th>Share</th></tr>{model_rows}</table></div>
  <div class="card"><h2>Task Distribution</h2><table><tr><th>Task</th><th>Requests</th><th>Top Model</th></tr>{task_rows}</table></div>
</div>
<div class="card" style="margin-top:16px"><h2>Recent Decisions</h2><table><tr><th>Time</th><th>Task</th><th>Model</th><th>Latency</th><th>Cost</th></tr>{recent_rows}</table></div>
<footer>clawRouter v2 · Auto-refreshes every 30s · <a href="/health" style="color:#58a6ff">API Health</a></footer>
</body></html>"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    total = _stats["total"] or 1
    actual = _cost_tracker["total_actual"]
    opus = _cost_tracker["total_if_opus"]
    saved = opus - actual
    pct = (saved / opus * 100) if opus > 0 else 0
    learned_pairs = sum(len(t) for t in _learned_scores.values())

    # Model usage from in-memory stats
    model_counts = defaultdict(int)
    task_counts = defaultdict(lambda: defaultdict(int))
    for model, tasks in _log_stats.items():
        for task, s in tasks.items():
            model_counts[model] += s["count"]
            task_counts[task][model] += s["count"]

    model_rows = ""
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct_m = c / total * 100
        model_rows += f"<tr><td>{m}</td><td>{c}</td><td><div class='bar'><div class='bar-fill' style='width:{pct_m}%'></div></div></td></tr>"

    task_rows = ""
    for t, models in sorted(task_counts.items(), key=lambda x: -sum(x[1].values())):
        tc = sum(models.values())
        top = max(models, key=models.get) if models else "-"
        task_rows += f"<tr><td><span class='tag tag-{t if t in ('code','chat','reason') else 'other'}'>{t}</span></td><td>{tc}</td><td>{top}</td></tr>"

    # Recent logs
    recent_rows = ""
    try:
        logfile = _log_path()
        if logfile.exists():
            with open(logfile) as f:
                lines = f.readlines()[-10:]
            for line in reversed(lines):
                try:
                    e = json.loads(line)
                    ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
                    recent_rows += f"<tr><td>{ts}</td><td><span class='tag tag-{e.get('task','') if e.get('task','') in ('code','chat','reason') else 'other'}'>{e.get('task','?')}</span></td><td>{e.get('model','?')}</td><td>{e.get('latency_ms','?')}ms</td><td>${e.get('cost_actual',0):.6f}</td></tr>"
                except: continue
    except: pass

    return DASHBOARD_HTML.format(
        total=_stats["total"], routed=_stats["routed"], passthrough=_stats["passthrough"],
        actual=actual, opus=opus, saved=saved, pct=pct, learned_pairs=learned_pairs,
        model_rows=model_rows or "<tr><td colspan='3'>No data yet</td></tr>",
        task_rows=task_rows or "<tr><td colspan='3'>No data yet</td></tr>",
        recent_rows=recent_rows or "<tr><td colspan='5'>No requests yet</td></tr>",
    )

# ═══════════════════════════════════════════════════════════════════
# 9. API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════
@app.get("/v1/models")
async def list_models(req: Request):
    resp = await _client.get(f"{LITELLM}/v1/models", headers={"Authorization": req.headers.get("authorization","")})
    data = resp.json()
    auto_models = [{"id": m, "object": "model", "owned_by": "clawrouter"} for m in sorted(AUTO)]
    data["data"] = auto_models + data.get("data", [])
    return JSONResponse(data)

@app.get("/health")
async def health():
    try: litellm_ok = (await _client.get(f"{LITELLM}/health/liveliness", timeout=5)).status_code == 200
    except: litellm_ok = False
    return {
        "router": "ok", "litellm": "ok" if litellm_ok else "down",
        "stats": _stats, "cost": _cost_tracker,
        "learned_pairs": sum(len(t) for t in _learned_scores.values()),
    }

@app.post("/learn")
async def trigger_learn():
    """Manually trigger learning from logs."""
    learn_from_logs()
    return {"status": "ok", "learned": {m: list(t.keys()) for m, t in _learned_scores.items()}}

@app.get("/test")
async def test():
    samples = [
        ("Hello, how are you?", "chat"),
        ("Write a quicksort in Python", "code"),
        ("Fix this TypeError in my code", "review"),
        ("```python\ndef foo():\n  return bar\n```\nThis crashes, why?", "review"),
        ("Explain microservices vs monolith step by step", "reason"),
        ("Write a blog post about AI trends", "creative"),
        ("Summarize this research paper", "summary"),
        ("Extract all emails from this text", "extract"),
        ("Calculate compound interest at 5%", "math"),
        ("Translate this to Spanish", "translate"),
        ("Design a production-ready REST API with auth, rate limiting, and caching", "code"),
        ("Compare PostgreSQL vs MongoDB for a real-time analytics dashboard", "reason"),
    ]
    results = []
    for prompt, expected in samples:
        msgs = [{"role": "user", "content": prompt}]
        feat = extract_features(msgs)
        task, cx = classify(feat, prompt)
        r = select(task, cx, feat, _cfg)
        results.append({"prompt": prompt[:60], "got": task, "expected": expected,
                        "match": task == expected, "model": r.model, "savings": r.savings})
    acc = sum(r["match"] for r in results)
    return {"accuracy": f"{acc}/{len(results)}", "results": results}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_all(req: Request, path: str):
    resp = await _client.request(req.method, f"{LITELLM}/{path}", content=await req.body(),
        headers={"Authorization": req.headers.get("authorization",""),
                 "Content-Type": req.headers.get("content-type", "application/json")})
    return Response(content=resp.content, status_code=resp.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4001)
