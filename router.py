"""
dkClaw Router — smart model selection proxy for LiteLLM.
Single file. Send model="auto" → router picks the cheapest model that handles your task well.

Client (port 4001) → Router → LiteLLM (port 4000) → Providers

Presets: auto (balanced) | auto-quality | auto-cheap | auto-fast
Any other model name passes through untouched.
"""

import re, os, json, time, logging, yaml
from dataclasses import dataclass
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

LITELLM = os.getenv("LITELLM_URL", "http://litellm:4000")
CFG_PATH = os.getenv("ROUTER_CONFIG", "/app/router_config.yaml")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [router] %(message)s")
log = logging.getLogger("router")

# ═══════════════════════════════════════════════════════
# MODELS — quality scores (0-1), costs, context windows
# Tune quality scores after testing on YOUR workloads.
# ═══════════════════════════════════════════════════════
MODELS = {
    #                          tier  $/1M-in  $/1M-out  ctx      speed  code  review reason creative chat  summary extract math  translate
    "claude-opus":    {"tier":1, "ci":15.0,  "co":75.0,  "ctx":200000, "spd":0.30, "q":{"code":0.97,"review":0.98,"reason":0.99,"creative":0.95,"chat":0.90,"summary":0.92,"extract":0.90,"math":0.95,"translate":0.90}},
    "claude-sonnet":  {"tier":1, "ci":3.0,   "co":15.0,  "ctx":200000, "spd":0.50, "q":{"code":0.95,"review":0.93,"reason":0.95,"creative":0.92,"chat":0.88,"summary":0.90,"extract":0.88,"math":0.90,"translate":0.88}},
    "claude-haiku":   {"tier":1, "ci":0.80,  "co":4.0,   "ctx":200000, "spd":0.85, "q":{"code":0.80,"review":0.75,"reason":0.75,"creative":0.80,"chat":0.85,"summary":0.85,"extract":0.85,"math":0.75,"translate":0.82}},
    "deepseek-chat":  {"tier":2, "ci":0.14,  "co":0.28,  "ctx":64000,  "spd":0.75, "q":{"code":0.90,"review":0.85,"reason":0.80,"creative":0.75,"chat":0.85,"summary":0.82,"extract":0.80,"math":0.85,"translate":0.78}},
    "deepseek-coder": {"tier":2, "ci":0.14,  "co":0.28,  "ctx":64000,  "spd":0.75, "q":{"code":0.93,"review":0.90,"reason":0.70,"creative":0.55,"chat":0.65,"summary":0.60,"extract":0.75,"math":0.80,"translate":0.50}},
    "gemini-flash":   {"tier":2, "ci":0.075, "co":0.30,  "ctx":1000000,"spd":0.95, "q":{"code":0.75,"review":0.70,"reason":0.72,"creative":0.75,"chat":0.82,"summary":0.85,"extract":0.82,"math":0.78,"translate":0.80}},
    "ollama-llama":   {"tier":3, "ci":0.0,   "co":0.0,   "ctx":8192,   "spd":0.60, "q":{"code":0.55,"review":0.45,"reason":0.40,"creative":0.50,"chat":0.70,"summary":0.55,"extract":0.50,"math":0.35,"translate":0.45}},
    "ollama-mistral": {"tier":3, "ci":0.0,   "co":0.0,   "ctx":8192,   "spd":0.60, "q":{"code":0.50,"review":0.45,"reason":0.40,"creative":0.55,"chat":0.70,"summary":0.55,"extract":0.50,"math":0.35,"translate":0.50}},
}

# ═══════════════════════════════════════════════════════
# CLASSIFIER — pattern → task type
# ═══════════════════════════════════════════════════════
PATTERNS = [
    (r"```",                                                                               "code",    3.0),
    (r"\b(implement|build|create|write|code)\b.*\b(function|class|api|endpoint|component|module|script|app|program|algorithm|sort|server)\b", "code", 2.5),
    (r"\b(write|implement|create|build|code)\b.*\bin (python|javascript|typescript|java|rust|go|c\+\+|ruby|php)\b", "code", 2.5),
    (r"\b(function|class|method|variable|loop|array|dict|list|regex|sql|html|css|json|yaml)\b", "code", 1.0),
    (r"\b(fix|debug|bug|error|traceback|exception|crash|broken|failing|TypeError|ValueError)\b", "review",  2.5),
    (r"\b(refactor|optimize|clean.?up|lint|migrate)\b.*\b(code|function|class)\b",         "review",  2.0),
    (r"\b(explain|why|how.does|analyze|compare|evaluate|pros?.and?.cons|trade.?off)\b",    "reason",  1.5),
    (r"\b(step.by.step|think.through|reason|logic|argument|critique|architecture)\b",      "reason",  2.0),
    (r"\b(calculate|compute|formula|equation|statistic|probability|average|median|percent)\b","math",   2.0),
    (r"\b(data.analy|chart|graph|plot|regression|dataset|csv|aggregate)\b",                "math",    1.5),
    (r"\b(write|draft|compose)\b.*\b(story|blog|essay|poem|article|post|email|letter)\b",  "creative",2.0),
    (r"\b(creative|imagine|brainstorm|narrative|fiction|copywriting)\b",                    "creative",1.5),
    (r"\b(summar\w*|tldr|tl;dr|condense|key.?points|digest|recap|gist)\b",                "summary", 3.0),
    (r"\b(extract|parse|find.all|list.all|pull.out|scrape|identify|detect|categorize)\b",  "extract", 2.0),
    (r"\b(translate|translation|translat)\b",                                              "translate",3.0),
    (r"\bin (spanish|french|german|chinese|japanese|hindi|arabic|korean|portuguese)\b",    "translate",2.0),
]

COMPLEXITY_BOOSTS = [
    (r"\b(step.by.step|detailed|comprehensive|thorough|in.depth)\b", 0.25),
    (r"\b(multiple|several|entire|complete|full)\b",                 0.15),
    (r"\b(production|enterprise|scalab|secure|robust)\b",            0.20),
    (r"```[\s\S]{500,}",                                             0.30),
]

PRESETS = {
    "balanced": (0.35, 0.45, 0.20),  # (quality, cost, speed)
    "quality":  (0.70, 0.10, 0.20),
    "cheap":    (0.15, 0.70, 0.15),
    "fast":     (0.15, 0.15, 0.70),
}

@dataclass
class Route:
    task: str; complexity: float; model: str; score: float; runner_up: str; savings: str

def classify_and_select(messages: list[dict], cfg: dict, preset: str = "balanced") -> Route:
    """Classify task from messages, score all models, return best pick."""
    text = " ".join(m.get("content","") for m in messages if m.get("role") in ("user","system") and isinstance(m.get("content"), str)).lower()
    tokens = max(len(text) // 4, 1)

    # ── Classify ──────────────────────────────────────
    scores = {}
    for pat, task, w in PATTERNS:
        hits = len(re.findall(pat, text))
        if hits: scores[task] = scores.get(task, 0) + hits * w

    task = max(scores, key=scores.get) if scores else ("chat" if tokens < 200 else "reason")
    if tokens < 8 and (max(scores.values()) if scores else 0) < 2.0 and task != "translate":
        task = "chat"

    cx = min(0.5, tokens / 8000)
    for pat, boost in COMPLEXITY_BOOSTS:
        if re.search(pat, text): cx += boost
    cx = round(min(1.0, cx), 3)

    long_ctx = tokens > 6000 or bool(re.search(r"\b(entire|whole|full)\b.{0,30}\b(file|document|code|repo)\b", text))

    # ── Select ────────────────────────────────────────
    wq, wc, ws = PRESETS.get(preset, PRESETS["balanced"])
    min_q = cfg.get("min_quality", 0.50)
    if cx > 0.7: min_q = max(min_q, 0.75)
    elif cx > 0.4: min_q = max(min_q, 0.60)

    overrides = cfg.get("overrides", {})
    if task in overrides and overrides[task]:
        return Route(task, cx, overrides[task], 1.0, "-", "override")

    bg = cfg.get("budget_guard", {})
    allowed = set(bg["cheap_models"]) if bg.get("enabled") and bg.get("force_cheap") else None

    max_cost = max((m["ci"]+m["co"])/2 for m in MODELS.values()) or 1
    candidates = []
    for name, m in MODELS.items():
        if allowed and name not in allowed: continue
        q = m["q"].get(task, 0.5)
        if q < min_q: continue
        if tokens > int(m["ctx"] * 0.8): continue
        if long_ctx and m["ctx"] < 32000: continue
        cost_s = 1.0 - ((m["ci"]+m["co"])/2) / max_cost
        final = q * wq + cost_s * wc + m["spd"] * ws
        candidates.append((round(final, 4), name))

    if not candidates:
        return Route(task, cx, "claude-haiku", 0.0, "-", "no-match-fallback")

    candidates.sort(reverse=True)
    winner = candidates[0]
    runner = candidates[1][1] if len(candidates) > 1 else "-"
    wm = MODELS[winner[1]]
    opus = MODELS["claude-opus"]
    opus_cost = (opus["ci"]+opus["co"])/2
    my_cost = (wm["ci"]+wm["co"])/2
    savings = f"{round((1 - my_cost/opus_cost)*100)}% cheaper than opus" if my_cost < opus_cost else "premium"

    return Route(task, cx, winner[1], winner[0], runner, savings)

# ═══════════════════════════════════════════════════════
# SERVER
# ═══════════════════════════════════════════════════════
_cfg = {}
_client = httpx.AsyncClient(timeout=120.0)
_stats = {"total": 0, "routed": 0, "passthrough": 0}

@asynccontextmanager
async def lifespan(_):
    global _cfg
    try:
        with open(CFG_PATH) as f: _cfg.update(yaml.safe_load(f) or {})
    except FileNotFoundError: pass
    log.info(f"Router ready → {LITELLM}")
    yield

app = FastAPI(title="dkClaw Router", lifespan=lifespan)
AUTO = {"auto","auto-quality","auto-cheap","auto-fast"}

@app.post("/v1/chat/completions")
async def route(req: Request):
    body = await req.json()
    auth = req.headers.get("authorization", "")
    model = body.get("model", "auto")
    _stats["total"] += 1

    if model in AUTO:
        _stats["routed"] += 1
        preset = model.split("-",1)[1] if "-" in model else "balanced"
        r = classify_and_select(body.get("messages", []), _cfg, preset)
        body["model"] = r.model
        meta = {"routed":True, "task":r.task, "cx":r.complexity, "model":r.model, "score":r.score, "runner_up":r.runner_up, "savings":r.savings}
        log.info(f"{r.task}|cx={r.complexity} → {r.model} ({r.savings})")
    else:
        _stats["passthrough"] += 1
        meta = {"routed":False, "model":model}

    headers = {"Authorization": auth, "Content-Type": "application/json"}
    try:
        if body.get("stream"):
            r = _client.build_request("POST", f"{LITELLM}/v1/chat/completions", json=body, headers=headers)
            resp = await _client.send(r, stream=True)
            async def stream():
                try:
                    async for chunk in resp.aiter_bytes(): yield chunk
                finally: await resp.aclose()
            return StreamingResponse(stream(), status_code=resp.status_code, media_type="text/event-stream", headers={"X-Router-Model": meta.get("model","")})
        else:
            t0 = time.monotonic()
            resp = await _client.post(f"{LITELLM}/v1/chat/completions", json=body, headers=headers)
            data = resp.json(); data["_router"] = {**meta, "ms": round((time.monotonic()-t0)*1000)}
            return JSONResponse(data, status_code=resp.status_code, headers={"X-Router-Model": meta.get("model","")})
    except httpx.HTTPError as e:
        return JSONResponse({"error": str(e)}, status_code=502)

@app.get("/v1/models")
async def models(req: Request):
    resp = await _client.get(f"{LITELLM}/v1/models", headers={"Authorization": req.headers.get("authorization","")})
    data = resp.json()
    data["data"] = [{"id":m,"object":"model","owned_by":"dkclaw-router"} for m in AUTO] + data.get("data",[])
    return JSONResponse(data)

@app.get("/health")
async def health():
    try: ok = (await _client.get(f"{LITELLM}/health/liveliness", timeout=5)).status_code == 200
    except: ok = False
    return {"router":"ok", "litellm":"ok" if ok else "down", "stats":_stats}

@app.get("/test")
async def test():
    samples = [
        ("Hello, how are you?", "chat"), ("Write a quicksort in Python", "code"),
        ("Fix this TypeError in my code", "review"), ("Explain microservices vs monolith step by step", "reason"),
        ("Write a blog post about AI trends", "creative"), ("Summarize this research paper", "summary"),
        ("Extract all emails from this text", "extract"), ("Calculate compound interest at 5%", "math"),
        ("Translate this to Spanish", "translate"),
    ]
    results = []
    for prompt, expected in samples:
        r = classify_and_select([{"role":"user","content":prompt}], _cfg)
        results.append({"prompt":prompt, "got":r.task, "expected":expected, "match":r.task==expected, "model":r.model, "savings":r.savings})
    return {"accuracy":f"{sum(r['match'] for r in results)}/{len(results)}", "results":results}

@app.api_route("/{path:path}", methods=["GET","POST","PUT","DELETE","PATCH"])
async def proxy(req: Request, path: str):
    resp = await _client.request(req.method, f"{LITELLM}/{path}", content=await req.body(),
        headers={"Authorization": req.headers.get("authorization",""), "Content-Type": req.headers.get("content-type","application/json")})
    return Response(content=resp.content, status_code=resp.status_code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4001)
