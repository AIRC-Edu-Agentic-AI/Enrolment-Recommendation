"""
Agentic router. Instead of a fixed parse->if/elif dispatch, Claude (Haiku) decides
which tools to call to answer an advising question. Tools fall into three roles:

  observe : gather VERIFIED facts (requirements, eligibility, audit, recommendation)
  infer   : compute plan changes (what-if simulation, plan comparison)
  display : finalize the reply (plain text, a plan-change branch, an infeasible note,
            or a clarifying question when the request is too ambiguous to act on)

Every tool is deterministic Python; the model never invents prerequisites, credits,
or feasibility — it only orchestrates and writes prose grounded in tool output. The
display tool the model chooses determines the API response shape (text vs. branch).

If the agent loop errors out, qa.handle falls back to the deterministic router, so
behaviour degrades gracefully.
"""
import json

import llm_client
from vprs import describe_prereq, term_label
from verifier import is_eligible
from planner import plan_path
from diff import diff_plans
import qa

MAX_STEPS = 6  # observe/infer turns before we force a wrap-up


# ---------------------------------------------------------------------------
# Tool schemas (grouped by role for the model)
# ---------------------------------------------------------------------------
TOOLS = [
    # --- observe ---
    {
        "name": "lookup_requirements",
        "description": "OBSERVE: get the prerequisite rule, credits, category, and "
                       "offering term for a course. Use for 'what are the prereqs of X', "
                       "'how many credits is X', 'when is X offered'.",
        "input_schema": {
            "type": "object",
            "properties": {"course": {"type": "string",
                                      "description": "Course code, e.g. CS401"}},
            "required": ["course"],
        },
    },
    {
        "name": "check_eligibility",
        "description": "OBSERVE: check whether the student can take a course right now "
                       "against their completed courses. Returns eligible/not plus the "
                       "reason. Use for 'can I take X', 'why can't I take X'.",
        "input_schema": {
            "type": "object",
            "properties": {"course": {"type": "string",
                                      "description": "Course code, e.g. CS301"}},
            "required": ["course"],
        },
    },
    {
        "name": "graduation_audit",
        "description": "OBSERVE: list the required courses still outstanding and any "
                       "category credit shortfalls. Use for 'what's left to graduate'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "recommend_courses",
        "description": "OBSERVE: the planner's recommended course set for the current "
                       "term. Use for 'what should I take next term'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # --- infer ---
    {
        "name": "simulate_plan_change",
        "description": "INFER: re-plan with a requested change and report the result "
                       "(feasible or not, where the course lands, forced downstream "
                       "changes, graduation impact). action is one of: "
                       "'pin' (take the course in a specific term — set 'term'; omit it "
                       "for the current term), 'delay', 'prefer_early' (take as early as "
                       "possible), 'drop'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["pin", "delay", "prefer_early", "drop"]},
                "course": {"type": "string", "description": "Course code"},
                "term": {"type": "integer",
                         "description": "Target term number for 'pin'/'delay' "
                                        "(optional; current term if omitted)"},
            },
            "required": ["action", "course"],
        },
    },
    {
        "name": "compare_plans",
        "description": "INFER: compare the current plan with the pending alternative "
                       "from the most recent simulate_plan_change. Use for 'compare the "
                       "two plans'.",
        "input_schema": {"type": "object", "properties": {}},
    },
    # --- display (terminal) ---
    {
        "name": "respond",
        "description": "DISPLAY: give the student a plain text answer. Use this to "
                       "finish when no plan branch is involved. Text must only state "
                       "facts returned by observe/infer tools.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "present_plan_change",
        "description": "DISPLAY: finish by showing the pending alternative plan as a "
                       "branch (from the most recent feasible simulate_plan_change). "
                       "Provide a short grounded summary as 'text'.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "report_infeasible",
        "description": "DISPLAY: finish by telling the student a requested change is not "
                       "possible, with the reason from simulate_plan_change. No branch "
                       "is shown.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "ask_clarification",
        "description": "ASK_USER: finish by asking the student a clarifying question when "
                       "the request is ambiguous and observing won't resolve it — e.g. you "
                       "cannot tell which course is meant, or a required detail (such as "
                       "the target term) is missing. Prefer this over guessing.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]

# Tools that end the reasoning loop and produce the API response.
_TERMINAL_TOOLS = {"respond", "present_plan_change", "report_infeasible",
                   "ask_clarification"}


# ---------------------------------------------------------------------------
# Tool execution + memory interface, bound to one request's (prog, state, session)
# ---------------------------------------------------------------------------
class _Ctx:
    """State builder + memory interface for one turn. Exposes three memory layers
    to the tools and enforces what may be written:

      structural (read-only) : the program/curriculum spec  -> self.prog, self.state
      episodic   (read/write): per-session working state     -> self.session
                               (current_plan, candidate, candidate_requested)
      long-term  (read-only) : student preferences/soft constraints -> self.prefs

    Tools write ONLY episodic state (the pending candidate). Long-term memory is a
    read view here; there is no persistence backend yet, so writes to it are not
    performed — preferences can be seeded into session['preferences'] upstream.
    """

    def __init__(self, prog, state, session):
        self.prog = prog        # structural memory (curriculum DAG, rules, metadata)
        self.state = state      # structural: the student's immutable record
        self.session = session  # episodic memory (mutable working state)
        self.prefs = session.get("preferences") or {}  # long-term view (read-only)
        base = session.get("current_plan")
        if not base:
            base, _ = plan_path(prog, state)
            session["current_plan"] = base
        self.base = base
        self.last_sim = None  # most recent simulate result (dict from qa.simulate_change)

    # -- helpers --
    def _code(self, raw):
        """Resolve a course code or name to a known code, or None."""
        if not raw:
            return None
        if raw in self.prog.subjects:
            return raw
        up = raw.strip().upper()
        if up in self.prog.subjects:
            return up
        low = raw.strip().lower()
        for c, s in self.prog.subjects.items():
            if s.name.lower() == low:
                return c
        return None

    # -- observe --
    def lookup_requirements(self, course):
        c = self._code(course)
        if not c:
            return {"error": f"unknown course '{course}'"}
        s = self.prog.subjects[c]
        return {"code": c, "name": s.name,
                "prerequisite": describe_prereq(s.prereq, self.prog.names()),
                "credits": s.credits, "category": s.category,
                "offered": s.offered, "source": s.source}

    def check_eligibility(self, course):
        c = self._code(course)
        if not c:
            return {"error": f"unknown course '{course}'"}
        ok, reason, src = is_eligible(self.prog, self.state, c)
        return {"code": c, "name": self.prog.subjects[c].name,
                "eligible": ok, "reason": reason, "source": src}

    def graduation_audit(self):
        ans = qa._resolve_audit(self.prog, self.state)
        return {"remaining": [{"code": c, "name": n} for c, n in ans["remaining"]],
                "categories_short": ans["categories_short"] or "none"}

    def recommend_courses(self):
        ans = qa._resolve_recommend(self.prog, self.state, self.session)
        return {"term": ans["term_label"],
                "subjects": [{"code": c, "name": n} for c, n in ans["subjects"]],
                "credits": ans["credits"]}

    # -- infer --
    def simulate_plan_change(self, action, course, term=None):
        c = self._code(course)
        if not c:
            return {"error": f"unknown course '{course}'"}
        sim = qa.simulate_change(self.prog, self.state, self.base,
                                 {"type": action, "code": c, "term": term})
        self.last_sim = sim
        if not sim["feasible"]:
            return {"feasible": False, "reason": sim["reason"]}
        return {"feasible": True, "summary": sim["summary"],
                "placed_term": term_label(sim["placed"]) if sim["placed"] else None,
                "requested_changes": sim["diff"]["requested_changes"],
                "induced_changes": sim["diff"]["induced_changes"],
                "graduation_term": term_label(sim["diff"]["grad_term_b"])
                if sim["diff"]["grad_term_b"] else None}

    def compare_plans(self):
        cand = self.session.get("candidate") or (
            self.last_sim["candidate"] if self.last_sim and self.last_sim["feasible"]
            else None)
        if not cand:
            return {"error": "no alternative plan to compare yet; simulate a change first"}
        d = diff_plans(self.prog, self.state, self.base, cand,
                       set(self.session.get("candidate_requested", [])))
        return {"summary": qa._summarize_diff(self.prog, d),
                "requested_changes": d["requested_changes"],
                "induced_changes": d["induced_changes"]}

    def dispatch(self, name, args):
        fn = getattr(self, name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:
            return {"error": str(e)}

    # -- display (build the API envelope) --
    def finalize(self, name, args):
        text = (args.get("text") or "").strip()
        if name == "present_plan_change":
            sim = self.last_sim
            if not sim or not sim["feasible"]:
                # nothing valid to show -> degrade to a plain answer
                return _envelope(text or "No alternative plan is available.")
            self.session["candidate"] = sim["candidate"]
            self.session["candidate_requested"] = list(sim["requested"])
            return _envelope(text or sim["summary"],
                             structured={"kind": "what_if", "summary": sim["summary"]},
                             candidate=qa._plan_view(self.prog, sim["candidate"]),
                             diff=sim["diff"])
        if name == "report_infeasible":
            reason = (self.last_sim or {}).get("reason")
            return _envelope(text or reason or "That isn't possible.",
                             structured={"kind": "infeasible",
                                         "reason": reason or text},
                             infeasible=True)
        if name == "ask_clarification":
            question = (args.get("question") or text).strip()
            env = _envelope(question or "Could you clarify what you'd like to do?",
                            structured={"kind": "clarify", "question": question})
            env["needs_input"] = True
            return env
        # respond
        return _envelope(text or "Done.")


def _envelope(text, structured=None, candidate=None, diff=None, infeasible=False):
    env = {"text": text, "structured": structured or {"kind": "agent"},
           "intent": "agent", "parse_source": "agent",
           "candidate": candidate, "diff": diff}
    if infeasible:
        env["infeasible"] = True
    return env


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def _system_prompt(prog, state, prefs=None):
    catalog = "\n".join(
        f"  {c}: {s.name} [{s.category}, {s.credits}cr, offered {s.offered}]"
        for c, s in prog.subjects.items())
    passed = ", ".join(sorted(state.passed_marks().keys())) or "none"
    prefs_line = ""
    if prefs:
        prefs_line = ("\nKnown student preferences (soft constraints): "
                      + "; ".join(f"{k}={v}" for k, v in prefs.items()) + ".")
    return f"""You are an academic-advising agent for {prog.name}. The student is \
{state.name}; the current term is term {state.current_term} ({term_label(state.current_term)}).

Answer the student's question by calling tools. Choose each step:
1. OBSERVE — call observe tools to gather verified facts (requirements, eligibility,
   graduation audit, recommendation). NEVER state a prerequisite, credit count, or
   eligibility result you did not get from a tool.
2. INFER — for "what if I…" / "I want to take X" / "compare plans", call
   simulate_plan_change or compare_plans.
3. Finish by calling exactly ONE terminal tool:
   - present_plan_change when a feasible plan change should be shown as a branch,
   - report_infeasible when a requested change is impossible,
   - respond for everything else (lookups, eligibility, audits, recommendations),
   - ask_clarification when the request is too ambiguous to act on (don't guess).

Map course names to codes using this catalog (use the codes when calling tools):
{catalog}

Courses already completed: {passed}.{prefs_line}

Keep replies brief (1-3 sentences), grounded only in tool results. Use plain prose
in sentence case — no markdown, bold, bullets, or headings.

CRITICAL — copy facts from tool results VERBATIM. Repeat term labels exactly as the
tool gives them (e.g. "Y2 Spring"); the only seasons are Fall and Spring (never
"Summer"). Repeat course codes, credit counts, and "need N more" figures exactly as
returned. Do NOT compute term numbers, invent seasons, recalculate or round credits,
or restate a number the tools did not give you. Resolve "this term"/"this semester"/
"now" to term {state.current_term} only when calling tools."""


def run(prog, state, session, question):
    """Drive the tool loop. Returns the API response envelope. Raises on hard failure
    (qa.handle catches and falls back to the deterministic router)."""
    ctx = _Ctx(prog, state, session)
    system = _system_prompt(prog, state, ctx.prefs)
    # Replay prior turns (plain user/assistant text) as conversation memory so the
    # agent can resolve references like "take it earlier" or "compare them". The
    # current turn's tool_use/tool_result messages stay local to this loop; only the
    # final answer is persisted to session['history'] (by qa.handle).
    prior = list(session.get("history", []))
    messages = prior + [{"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        resp = llm_client.chat_tools(system, messages, TOOLS, max_tokens=700)
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            # model answered with plain text and no tool -> treat as a respond()
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return _envelope(text or "Done.")

        # A terminal tool ends the turn. If several tools came back, honour the
        # terminal one and ignore the rest.
        display = next((t for t in tool_uses if t.name in _TERMINAL_TOOLS), None)
        if display is not None:
            return ctx.finalize(display.name, display.input)

        # Otherwise run the observe/infer tools and feed results back.
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for t in tool_uses:
            out = ctx.dispatch(t.name, t.input)
            results.append({"type": "tool_result", "tool_use_id": t.id,
                            "content": json.dumps(out, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})

    # Ran out of steps without a terminal tool -> force a grounded wrap-up.
    messages.append({"role": "user",
                     "content": "Now call exactly one terminal tool to finish."})
    resp = llm_client.chat_tools(system, messages, TOOLS, max_tokens=400,
                                 tool_choice={"type": "any"})
    display = next((b for b in resp.content
                    if b.type == "tool_use" and b.name in _TERMINAL_TOOLS), None)
    if display is not None:
        return ctx.finalize(display.name, display.input)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return _envelope(text or "I couldn't complete that. Please rephrase.")
