"""
LLM client. A small model does exactly two jobs:

  parse(question)  -> typed query  {intent, subjects, action}
  render(answer)   -> grounded prose, constrained to supplied facts

Everything regulatory/planning is computed deterministically elsewhere. A small
model will confabulate prerequisites and miscount credits, so it never produces
answer content.

The backend is selected with the LLM_PROVIDER env variable (see .env.example):

  LLM_PROVIDER=anthropic   -> Claude via the official Anthropic SDK (default; uses Haiku)
  LLM_PROVIDER=lmstudio    -> a local OpenAI-compatible endpoint (LM Studio)

If the chosen backend is unreachable (or returns junk), both functions fall back
to a deterministic path, so the app stays fully usable without the model.
"""
import json
import re
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()

# --- Anthropic (Claude) backend -------------------------------------------
# Default model is Haiku; override with ANTHROPIC_MODEL. The API key is read
# from ANTHROPIC_API_KEY by the SDK.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

# --- LM Studio (OpenAI-compatible) backend --------------------------------
LM_BASE = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LM_MODEL = os.environ.get("LMSTUDIO_MODEL", "local-model")  # LM Studio ignores/maps this
TIMEOUT = float(os.environ.get("LMSTUDIO_TIMEOUT", "30"))
CONNECT_TIMEOUT = float(os.environ.get("LMSTUDIO_CONNECT_TIMEOUT", "2"))

INTENTS = ["regulatory_lookup", "eligibility", "why_not", "recommend",
           "what_if", "graduation_audit", "out_of_scope"]

# Cached availability so a request never hangs probing an absent model.
# When the backend is offline we remember that for AVAIL_TTL seconds; parse/render
# then go straight to the deterministic fallback without attempting a call.
_avail_cache = {"ok": None, "ts": 0.0}
AVAIL_TTL = float(os.environ.get("LLM_AVAIL_TTL", os.environ.get("LMSTUDIO_AVAIL_TTL", "10")))


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------
_anthropic_client = None


def _get_anthropic():
    """Lazily construct the Anthropic client. Returns None if unusable
    (SDK missing or ANTHROPIC_API_KEY unset)."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    except Exception:
        return None
    return _anthropic_client


def _split_system(messages):
    """Anthropic takes the system prompt as a top-level arg, not a message role.
    Pull any system turns out and return (system_text, [user/assistant turns])."""
    system_parts, convo = [], []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    return "\n\n".join(system_parts), convo


def _chat_anthropic(messages, temperature, max_tokens):
    client = _get_anthropic()
    if client is None:
        raise RuntimeError("Anthropic client unavailable")
    system_text, convo = _split_system(messages)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_text,
        messages=convo,
    )
    # Concatenate text blocks; ignore any non-text content.
    return "".join(b.text for b in resp.content if b.type == "text")


# ---------------------------------------------------------------------------
# LM Studio backend (OpenAI-compatible)
# ---------------------------------------------------------------------------
# A session that ignores proxy/env settings. LM Studio is a local endpoint, so a
# stray HTTP_PROXY in the environment must never route (and stall) these calls.
_session = None


def _get_session():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.trust_env = False
    return _session


def _chat_lmstudio(messages, temperature, max_tokens):
    """Raw call to LM Studio's OpenAI-compatible endpoint. Raises on failure.

    Uses a (connect, read) timeout tuple so an absent endpoint fails fast on
    connect instead of blocking on the full read timeout.
    """
    r = _get_session().post(
        f"{LM_BASE}/chat/completions",
        json={"model": LM_MODEL, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens, "stream": False},
        timeout=(CONNECT_TIMEOUT, TIMEOUT))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Transport dispatch
# ---------------------------------------------------------------------------
def _chat(messages, temperature=0.0, max_tokens=1024):
    """Send an OpenAI-style message list to the configured backend. Raises on failure."""
    if PROVIDER == "anthropic":
        return _chat_anthropic(messages, temperature, max_tokens)
    return _chat_lmstudio(messages, temperature, max_tokens)


def llm_available(force=False):
    """Cached health check; result cached for AVAIL_TTL seconds."""
    now = time.time()
    if not force and _avail_cache["ok"] is not None and now - _avail_cache["ts"] < AVAIL_TTL:
        return _avail_cache["ok"]
    if PROVIDER == "anthropic":
        # Treat a constructible client (SDK present + API key set) as online. A
        # genuinely bad key surfaces on the first call and falls back gracefully.
        ok = _get_anthropic() is not None
    else:
        try:
            _get_session().get(f"{LM_BASE}/models",
                               timeout=(CONNECT_TIMEOUT, 5)).raise_for_status()
            ok = True
        except Exception:
            ok = False
    _avail_cache["ok"] = ok
    _avail_cache["ts"] = now
    return ok


# ---------------------------------------------------------------------------
# Tool use (agentic routing)
# ---------------------------------------------------------------------------
# Tool use (agentic routing)
# ---------------------------------------------------------------------------
def agent_available():
    return llm_available()


# Fake Anthropic SDK response objects so agent.py needs no changes.
class _Block:
    __slots__ = ("type", "text", "id", "name", "input")

    def __init__(self, type, **kw):
        self.type = type
        self.text = kw.get("text", "")
        self.id = kw.get("id")
        self.name = kw.get("name")
        self.input = kw.get("input")


class _Resp:
    __slots__ = ("content",)

    def __init__(self, content):
        self.content = content


def _tools_to_openai(tools):
    """Anthropic input_schema -> OpenAI function parameters."""
    return [
        {"type": "function",
         "function": {
             "name": t["name"],
             "description": t.get("description", ""),
             "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
         }}
        for t in tools
    ]


def _messages_to_openai(system, messages):
    """Convert Anthropic-format message list to OpenAI format.

    Handles: plain string content, assistant blocks (text + tool_use),
    and user tool-result lists. Works on both raw dicts and Anthropic SDK
    objects (which agent.py stores back into the message list).
    """
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        # list of blocks
        if role == "user":
            tool_results = [c for c in content
                            if _block_type(c) == "tool_result"]
            text_parts = [c for c in content
                          if _block_type(c) == "text"]
            if text_parts:
                out.append({"role": "user",
                            "content": "\n".join(_block_text(c) for c in text_parts)})
            for tr in tool_results:
                tid = tr["tool_use_id"] if isinstance(tr, dict) else tr.tool_use_id
                body = tr["content"] if isinstance(tr, dict) else tr.content
                if not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False)
                out.append({"role": "tool", "tool_call_id": tid, "content": body})
        elif role == "assistant":
            text = "".join(_block_text(c) for c in content
                           if _block_type(c) == "text")
            tool_calls = []
            for c in content:
                if _block_type(c) == "tool_use":
                    cid = c["id"] if isinstance(c, dict) else c.id
                    name = c["name"] if isinstance(c, dict) else c.name
                    inp = c["input"] if isinstance(c, dict) else c.input
                    tool_calls.append({
                        "id": cid, "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(inp, ensure_ascii=False)},
                    })
            msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
    return out


def _block_type(b):
    return b.get("type") if isinstance(b, dict) else getattr(b, "type", None)


def _block_text(b):
    return b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")


def _chat_tools_lmstudio(system, messages, tools, max_tokens, temperature, tool_choice):
    """Tool-calling via OpenAI-compatible endpoint; returns a fake Anthropic response."""
    body = {
        "model": LM_MODEL,
        "messages": _messages_to_openai(system, messages),
        "tools": _tools_to_openai(tools),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tool_choice is not None:
        tc_type = tool_choice.get("type")
        body["tool_choice"] = "required" if tc_type == "any" else "auto"

    r = _get_session().post(f"{LM_BASE}/chat/completions", json=body,
                            timeout=(CONNECT_TIMEOUT, TIMEOUT))
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]

    blocks = []
    if msg.get("content"):
        blocks.append(_Block("text", text=msg["content"]))
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        try:
            inp = json.loads(fn["arguments"])
        except Exception:
            inp = {}
        blocks.append(_Block("tool_use", id=tc["id"], name=fn["name"], input=inp))
    return _Resp(blocks)


def chat_tools(system, messages, tools, max_tokens=2048, temperature=0.0,
               tool_choice=None):
    """One tool-use turn. Returns an Anthropic-compatible response object.
    Works for both the Anthropic and LM Studio / OpenAI-compatible backends."""
    if PROVIDER == "anthropic":
        client = _get_anthropic()
        if client is None:
            raise RuntimeError("Anthropic client unavailable")
        kwargs = dict(model=ANTHROPIC_MODEL, max_tokens=max_tokens,
                      temperature=temperature, system=system,
                      messages=messages, tools=tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return client.messages.create(**kwargs)
    return _chat_tools_lmstudio(system, messages, tools, max_tokens,
                                temperature, tool_choice)


# ---------------------------------------------------------------------------
# PARSE
# ---------------------------------------------------------------------------
_PARSE_SYS = """You convert a student's academic-advising question into a JSON query.
Questions may be in English or Vietnamese. Common Vietnamese intent signals:
- "sang kỳ", "vào kỳ", "dời sang", "chuyển sang", "học kỳ" -> what_if / pin
- "trì hoãn", "dời lại", "đẩy lùi" -> what_if / delay
- "học sớm", "càng sớm càng tốt" -> what_if / prefer_early
- "bỏ", "xóa" -> what_if / drop
- "có thể học", "đủ điều kiện" -> eligibility
- "tại sao không", "vì sao không" -> why_not
- "còn thiếu", "cần gì để tốt nghiệp" -> graduation_audit
Output ONLY a JSON object, no prose, no markdown fences.

Schema:
{
  "intent": one of ["regulatory_lookup","eligibility","why_not","recommend","what_if","graduation_audit","out_of_scope"],
  "subjects": [list of course codes mentioned, e.g. "CS301"],
  "action": null OR {"type": "delay"|"prefer_early"|"drop"|"pin", "code": "CSxxx", "term": int|null}
}

Intent guide:
- regulatory_lookup: asks what a rule/prerequisite/credit requirement IS.
- eligibility: "can I take X (now)?"
- why_not: "why can't I take X?"
- recommend: "what should I take next / next term?"
- what_if: asks to change the plan (move/delay/take earlier/drop a course, or put a
  course in a specific term) or compare plans.
- graduation_audit: "what's left to graduate?", remaining requirements.
- out_of_scope: subjective/advice unrelated to rules (is a professor good, should I change major).

Action types (only for what_if):
- pin: the student wants a course IN a specific term, e.g. "take X this semester",
  "I want X this term, what should I do", "put X in term 5", "schedule X next term".
  Set "term" to the term number when one is named or implied; for "this term/this
  semester/now" leave "term": null (the app resolves it to the current term).
- prefer_early: "take X as early as possible / sooner" with NO specific target term.
- delay: "delay / postpone / push X to later".
- drop: "drop / remove X".

Use the provided course list to map names to codes. If none apply, subjects=[]."""

_PARSE_EXAMPLES = [
    ("What are the prerequisites for Machine Learning?",
     '{"intent":"regulatory_lookup","subjects":["UET.CS3136"],"action":null}'),
    ("Can I take Artificial Intelligence now?",
     '{"intent":"eligibility","subjects":["UET.CS2046"],"action":null}'),
    ("Why can't I register for the Graduation Thesis?",
     '{"intent":"why_not","subjects":["UET.CS4023"],"action":null}'),
    ("What should I take next term?",
     '{"intent":"recommend","subjects":[],"action":null}'),
    ("I want to take Artificial Intelligence this semester, what should I do?",
     '{"intent":"what_if","subjects":["UET.CS2046"],"action":{"type":"pin","code":"UET.CS2046","term":null}}'),
    ("Put Machine Learning in term 5",
     '{"intent":"what_if","subjects":["UET.CS3136"],"action":{"type":"pin","code":"UET.CS3136","term":5}}'),
    ("I want to delay Artificial Intelligence",
     '{"intent":"what_if","subjects":["UET.CS2046"],"action":{"type":"delay","code":"UET.CS2046","term":null}}'),
    ("Take Machine Learning as early as possible",
     '{"intent":"what_if","subjects":["UET.CS3136"],"action":{"type":"prefer_early","code":"UET.CS3136","term":null}}'),
    ("Compare the two plans",
     '{"intent":"what_if","subjects":[],"action":null}'),
    ("What do I still need to graduate?",
     '{"intent":"graduation_audit","subjects":[],"action":null}'),
]


def parse_question(question: str, subject_index: dict, current_term=None):
    """subject_index: code -> name. Returns dict; always falls back gracefully."""
    if not llm_available():
        fb = _fallback_parse(question, subject_index, current_term)
        fb["_source"] = "fallback"
        return fb
    course_list = "\n".join(f"{c}: {n}" for c, n in subject_index.items())
    sys_prompt = _PARSE_SYS
    if current_term is not None:
        sys_prompt += (f"\n\nThe current term is term {current_term}. "
                       f'Map "this term"/"this semester"/"now" to it and '
                       f'"next term" to term {current_term + 1}.')
    msgs = [{"role": "system", "content": sys_prompt + "\n\nCourses:\n" + course_list}]
    for q, a in _PARSE_EXAMPLES:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    try:
        raw = _chat(msgs, max_tokens=1024)
        obj = _extract_json(raw)
        if obj and obj.get("intent") in INTENTS:
            obj.setdefault("subjects", [])
            obj.setdefault("action", None)
            obj["_source"] = "llm"
            return obj
    except Exception:
        pass
    fb = _fallback_parse(question, subject_index, current_term)
    fb["_source"] = "fallback"
    return fb


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _fallback_parse(question, subject_index, current_term=None):
    q = question.lower()
    # find subject codes and names (full code match)
    found = []
    for c in subject_index:
        if c.lower() in q:
            found.append(c)
    for c, n in subject_index.items():
        if n.lower() in q and c not in found:
            found.append(c)
    # partial numeric-suffix match: "2046" -> "UET.CS2046"
    if not found:
        for c in subject_index:
            suffix = re.sub(r'^[A-Za-z.]+', '', c)
            if suffix and re.search(rf'\b{re.escape(suffix)}\b', q):
                found.append(c)
    # distinctive-token match (e.g. "capstone" -> "Capstone Project")
    if not found:
        stop = {"to", "of", "and", "in", "i", "ii", "the", "for", "introduction"}
        for c, n in subject_index.items():
            toks = [t for t in re.findall(r"[a-z]+", n.lower())
                    if len(t) >= 4 and t not in stop]
            if any(re.search(rf"\b{re.escape(t)}\b", q) for t in toks):
                found.append(c)
    # intent by keyword
    action = None
    if any(w in q for w in ["why can", "why cant", "why can't", "why not"]):
        intent = "why_not"
    elif any(w in q for w in ["delay", "later", "push", "postpone"]):
        intent, action = "what_if", _mk_action("delay", found)
    elif any(w in q for w in ["earlier", "as early", "sooner", "asap", "advance"]):
        intent, action = "what_if", _mk_action("prefer_early", found)
    elif any(w in q for w in ["drop", "remove", "skip"]):
        intent, action = "what_if", _mk_action("drop", found)
    elif found and any(w in q for w in [
            "this semester", "this term", "in term", "next term", " now",
            "fit ", "schedule ", "put ",
            "sang kỳ", "vào kỳ", "dời sang", "chuyển sang", "học kỳ"]):
        # "take X this semester / in term N / next term" -> pin to that term
        intent, action = "what_if", _mk_action("pin", found, _term_ref(q, current_term))
    elif any(w in q for w in ["compare", "difference", "versus", " vs "]):
        intent = "what_if"
    elif any(w in q for w in ["retake", "repeat", "can i take", "am i eligible",
                               "eligible", "allowed to"]):
        intent = "eligibility"
    elif any(w in q for w in ["graduate", "left", "remaining", "still need"]):
        intent = "graduation_audit"
    elif any(w in q for w in ["recommend", "next term", "what should", "what do i take"]):
        intent = "recommend"
    elif any(w in q for w in ["prerequisite", "prereq", "require", "requirement",
                               "credits", "how many", "when is", "when can"]):
        intent = "regulatory_lookup"
    elif found:
        intent = "regulatory_lookup"
    else:
        intent = "out_of_scope"
    return {"intent": intent, "subjects": found, "action": action}


def _mk_action(typ, found, term=None):
    return {"type": typ, "code": found[0], "term": term} if found else None


def _term_ref(q, current_term):
    """Resolve a term reference in free text to a term number (or None)."""
    m = re.search(r"term\s+(\d+)", q)
    if m:
        return int(m.group(1))
    # "Y3 Fall" / "Y3 Spring" / "y3fall" etc.
    m = re.search(r"y(\d+)\s*(fall|spring)", q, re.IGNORECASE)
    if m:
        year, season = int(m.group(1)), m.group(2).lower()
        return (year - 1) * 2 + (1 if season == "fall" else 2)
    if current_term is not None and "next term" in q:
        return current_term + 1
    return None  # "this term"/"this semester"/"now" -> resolved downstream


# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------
_RENDER_SYS = """You are an academic-advising assistant. You are given a JSON object of
VERIFIED facts. Write a brief, plain reply (1-3 sentences) for a student using ONLY
those facts. Do not add prerequisites, numbers, course codes, or claims not present
in the JSON. Do not apologize. Sentence case, no markdown."""


def render(answer: dict) -> str:
    """Render a structured answer to prose. Falls back to a deterministic template."""
    if not llm_available():
        return template_render(answer)
    try:
        raw = _chat([{"role": "system", "content": _RENDER_SYS},
                     {"role": "user", "content": json.dumps(answer, ensure_ascii=False)}],
                    max_tokens=768)
        out = raw.strip()
        if out:
            return out
    except Exception:
        pass
    return template_render(answer)


def template_render(answer: dict) -> str:
    """Deterministic, always-correct rendering (used as fallback)."""
    k = answer.get("kind")
    if k == "prereq":
        return f"{answer['subject_name']} ({answer['subject']}) requires: {answer['prereq_text']}."
    if k == "eligibility":
        if answer["eligible"]:
            return f"Yes — you can register for {answer['subject']} ({answer['subject_name']}); {answer['reason']}."
        return f"No — {answer['subject']} ({answer['subject_name']}) is not available yet: {answer['reason']}."
    if k == "why_not":
        return f"You can't take {answer['subject']} ({answer['subject_name']}) yet because its {answer['reason']}. (Source: {answer.get('source','program catalog')})"
    if k == "recommend":
        items = ", ".join(f"{c} ({n})" for c, n in answer["subjects"])
        return f"For {answer['term_label']}, a valid set is: {items} ({answer['credits']} credits)."
    if k == "graduation_audit":
        if not answer["remaining"]:
            return "You have satisfied all requirements."
        rem = ", ".join(f"{c} ({n})" for c, n in answer["remaining"])
        return f"To graduate you still need: {rem}. Categories short: {answer.get('categories_short') or 'none'}."
    if k == "what_if":
        return answer["summary"]
    if k == "infeasible":
        return f"That isn't possible: {answer['reason']}"
    if k == "out_of_scope":
        return "That's outside what I can verify from the program rules. An academic advisor can help with that."
    return answer.get("summary", "Done.")
