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
import os

import llm_client
from vprs import describe_prereq, term_label
from verifier import is_eligible
from planner import plan_path
from diff import diff_plans
import qa

MAX_STEPS = 6  # observe/infer turns before we force a wrap-up

# P4: the deterministic guardrails (auto-finalize, terminal correction) compensate
# for a weak local model. They are LOAD-BEARING with the 9B model but cap how
# agentic the loop can be. Make them measurable and switchable so their cost is
# visible and a stronger model can run "trust mode" (guardrails off).
#   ATLAS_GUARDRAILS=off    -> don't auto-finalize or override the model's terminal
#   ATLAS_AGENT_DEBUG=1     -> attach an intervention trace to the response envelope
_GUARDRAILS = os.environ.get("ATLAS_GUARDRAILS", "on").lower() != "off"
_AGENT_DEBUG = os.environ.get("ATLAS_AGENT_DEBUG", "").lower() in ("1", "on", "true")


# ---------------------------------------------------------------------------
# Tool schemas (grouped by role for the model)
# ---------------------------------------------------------------------------
TOOLS = [
    # --- observe ---
    {
        "name": "search_course",
        "description": "OBSERVE: find courses by name, keyword, or partial code. "
                       "Call this first whenever the student mentions a course by name "
                       "or a partial code to get the exact course code for other tools.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "Course name, keyword, or partial "
                                                    "code, e.g. 'machine learning', "
                                                    "'CS204', 'database'"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_plan_overview",
        "description": "OBSERVE: get the current plan term-by-term: credit totals and "
                       "courses per future term. Call this for workload/balance questions "
                       "or whenever you need to see the full schedule.",
        "input_schema": {"type": "object", "properties": {}},
    },
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
        "name": "replan",
        "description": "INFER: re-plan the remaining terms under ANY combination of "
                       "constraints, and report the result (feasibility, where courses "
                       "land, forced downstream changes, graduation impact). This single "
                       "tool handles every plan change — pin/move a course, take one "
                       "earlier, delay one, drop one, raise the credit cap, force an "
                       "earlier or later graduation, or rebalance the workload — and you "
                       "may COMBINE them in one call (e.g. pin a course AND raise the cap "
                       "AND graduate later). Term references use labels like 'Y5 Fall'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pin": {
                    "type": "array",
                    "description": "Force courses into specific terms.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "course": {"type": "string"},
                            "term_label": {"type": "string",
                                           "description": "e.g. 'Y5 Fall', 'Y3 Spring'"},
                        },
                        "required": ["course", "term_label"],
                    },
                },
                "earliest": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Courses to schedule as early as possible "
                                   "('take X sooner').",
                },
                "delay": {
                    "type": "array",
                    "description": "Push a course to no earlier than the given term.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "course": {"type": "string"},
                            "term_label": {"type": "string"},
                        },
                        "required": ["course", "term_label"],
                    },
                },
                "drop": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Courses to remove from the plan.",
                },
                "credit_cap": {
                    "type": "integer",
                    "description": "Override the per-term credit limit (default 18). "
                                   "Set this only when the student explicitly asks to "
                                   "'take more credits' or names a cap. For 'graduate "
                                   "earlier' use graduate_earlier_by instead and let the "
                                   "system work out the cap.",
                },
                "graduate_earlier_by": {
                    "type": "integer",
                    "description": "Pull graduation EARLIER by this many terms: 1 for "
                                   "'a term/semester sooner', 2 for 'a year sooner'. "
                                   "Relative to the current graduation term — you do NOT "
                                   "need to know it, and do NOT guess a credit cap; if it "
                                   "needs a heavier load the system will offer that.",
                },
                "graduate_later_by": {
                    "type": "integer",
                    "description": "Push graduation LATER by this many terms: 1 for "
                                   "'one term/semester later', 2 for 'one year later'. "
                                   "Computed relative to the current graduation term — "
                                   "you do NOT need to know that term.",
                },
                "graduate_by": {
                    "type": "string",
                    "description": "Latest acceptable graduation term as a label, e.g. "
                                   "'Y4 Spring'. Use only when the student names a "
                                   "specific deadline.",
                },
                "objective": {
                    "type": "string",
                    "enum": ["early", "even", "minimal_change"],
                    "description": "What to optimize: 'early' = graduate ASAP; 'even' = "
                                   "spread credits evenly across terms (balance / lighten "
                                   "the workload); 'minimal_change' = keep the current "
                                   "plan as intact as possible (default for small edits).",
                },
            },
        },
    },
    {
        "name": "compare_plans",
        "description": "INFER: compare two plans. With no arguments, compares the current "
                       "plan against the most recent alternative. Pass candidate ids "
                       "(e.g. plan_a='c1', plan_b='c2', shown in the session context) to "
                       "compare two specific pending alternatives.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan_a": {"type": "string",
                           "description": "Candidate id, or omit for the current plan."},
                "plan_b": {"type": "string",
                           "description": "Candidate id, or omit for the latest alternative."},
            },
        },
    },
    # --- display (terminal) ---
    {
        "name": "respond",
        "description": "DISPLAY: give the student a plain text answer. Use ONLY for "
                       "lookups, eligibility checks, audits, and recommendations — "
                       "i.e. when you did NOT call replan. "
                       "Never use respond after a replan.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "present_plan_change",
        "description": "DISPLAY: REQUIRED after every feasible replan. "
                       "Shows the alternative plan as a branch the student can accept "
                       "or discard. Provide a 1-2 sentence grounded summary as 'text'.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "report_infeasible",
        "description": "DISPLAY: REQUIRED after an infeasible replan. "
                       "Tells the student the change is impossible and why.",
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
# These infer tools own the last_sim state and trigger auto-finalize
# so the model never has a second chance to make the wrong terminal call.
_PLAN_INFER_TOOLS = {"replan"}


# ---------------------------------------------------------------------------
# Tool execution + memory interface, bound to one request's (prog, state, session)
# ---------------------------------------------------------------------------
class _Ctx:
    """State builder + memory interface for one turn. Exposes three memory layers
    to the tools and enforces what may be written:

      structural (read-only) : the program/curriculum spec  -> self.prog, self.state
      episodic   (read/write): per-session working state     -> self.session
                               (current_plan, candidates[], last_constraints)
      long-term  (read-only) : student preferences/soft constraints -> self.prefs

    Tools write ONLY episodic state (the pending candidate branches). Long-term memory is a
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
        self.trace = []           # P4: scaffolding interventions this turn (observability)
        self.corrections_left = 1  # P4: budget for nudging the model past a bad terminal

    def note(self, event):
        """Record a scaffolding intervention (auto-finalize, terminal correction,
        tool error, self-correction) so its frequency is measurable."""
        self.trace.append(event)

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

    def search_course(self, query):
        q = query.lower().strip()
        results = []
        for c, s in self.prog.subjects.items():
            if q in c.lower() or q in s.name.lower():
                results.append({
                    "code": c, "name": s.name,
                    "credits": s.credits, "category": s.category,
                    "offered": s.offered,
                    "prerequisite": describe_prereq(s.prereq, self.prog.names()),
                })
        if not results:
            return {"error": f"no courses match '{query}'"}
        return results[:8]

    def get_plan_overview(self):
        cur = self.state.current_term
        terms = []
        for t in sorted(self.base):
            if t < cur:
                continue
            codes = self.base[t]
            cr = sum(self.prog.subjects[c].credits for c in codes
                     if c in self.prog.subjects)
            terms.append({
                "term": t, "label": term_label(t), "total_credits": cr,
                "courses": [{"code": c, "name": self.prog.subjects[c].name,
                             "credits": self.prog.subjects[c].credits}
                            for c in codes if c in self.prog.subjects],
            })
        return {"current_term": cur, "terms": terms}

    @staticmethod
    def _parse_term_label(label):
        """Convert 'Y5 Fall', 'Y3 Spring', 'Y5' → integer term number."""
        import re
        if label is None:
            return None
        if isinstance(label, int):
            return label
        s = str(label).strip()
        m = re.match(r"[Yy](\d+)\s*(fall|spring)?", s, re.IGNORECASE)
        if not m:
            return None
        year = int(m.group(1))
        season = (m.group(2) or "fall").lower()
        return (year - 1) * 2 + (1 if season == "fall" else 2)

    # -- infer --
    def replan(self, pin=None, earliest=None, delay=None, drop=None,
               credit_cap=None, graduate_by=None, graduate_earlier_by=None,
               graduate_later_by=None, graduate_no_earlier_than=None, objective=None):
        """The single plan-change capability. Assemble the requested natural-language
        intent into a constraint set, solve once, and return the diff. Every
        constraint composes — pins, delays, drops, credit cap, graduation bounds,
        and objective can all be supplied together. If the request is infeasible,
        search for minimal relaxations that would make it possible (P2)."""
        mods, requested = self._build_constraints(
            pin, earliest, delay, drop, credit_cap, graduate_by, graduate_earlier_by,
            graduate_later_by, graduate_no_earlier_than, objective)
        if "error" in mods:
            return {"feasible": False, "reason": mods["error"]}

        cand, reason = plan_path(self.prog, self.state,
                                 modifiers=mods, anchor=self.base)
        if reason:
            # Don't dead-end: look for minimal relaxations that make it feasible.
            alts = self._find_alternatives(mods)
            self.last_sim = {"feasible": False, "reason": reason, "alternatives": alts}
            out = {"feasible": False, "reason": reason}
            if alts:
                out["alternatives"] = [
                    {"description": a["description"],
                     "graduation_term": a["graduation_term"],
                     "peak_credits_per_term": a["peak_credits_per_term"]}
                    for a in alts]
            return out
        out = self._present(cand, requested, credit_cap=mods.get("credit_cap"))
        # Remember the constraints behind this candidate so the next turn can carry
        # the intent forward (e.g. refine a balance instead of starting over).
        self.last_sim["constraints"] = self._describe_constraints(mods)
        return out

    def _find_alternatives(self, mods):
        """For an infeasible constraint set, get minimal relaxations that work, each
        annotated with its graduation term and peak load for presentation."""
        from planner import relax, plan_credits, grad_term
        alts = []
        for r in relax(self.prog, self.state, self.base, mods):
            plan = r["plan"]
            future = [v for t, v in plan_credits(self.prog, plan).items()
                      if t >= self.state.current_term]
            gt = grad_term(plan)
            alts.append({
                "description": r["description"],
                "modifiers": r["modifiers"],
                "plan": plan,
                "graduation_term": term_label(gt) if gt else None,
                "peak_credits_per_term": max(future) if future else 0,
            })
        return alts

    @staticmethod
    def _describe_constraints(mods):
        """Render a constraint set as a short human-readable phrase for cross-turn
        continuity (stored in session and shown back to the agent next turn)."""
        parts = []
        if mods.get("objective"):
            parts.append(f"objective={mods['objective']}")
        if mods.get("credit_cap"):
            parts.append(f"credit cap {mods['credit_cap']}")
        if mods.get("min_grad_term"):
            parts.append(f"graduate no earlier than {term_label(mods['min_grad_term'])}")
        if mods.get("max_grad_term"):
            parts.append(f"graduate by {term_label(mods['max_grad_term'])}")
        if mods.get("pin"):
            parts.append("pinned " + ", ".join(
                f"{c}->{term_label(t)}" for c, t in mods["pin"].items()))
        if mods.get("not_before"):
            parts.append("delayed " + ", ".join(
                f"{c}->{term_label(t)}" for c, t in mods["not_before"].items()))
        if mods.get("prefer_early"):
            parts.append("earlier: " + ", ".join(sorted(mods["prefer_early"])))
        if mods.get("drop"):
            parts.append("dropped: " + ", ".join(sorted(mods["drop"])))
        return "; ".join(parts) or "no special constraints"

    def _build_constraints(self, pin, earliest, delay, drop, credit_cap,
                           graduate_by, graduate_earlier_by, graduate_later_by,
                           graduate_no_earlier_than, objective):
        """Translate the tool's natural-language-shaped args into planner modifiers.
        Returns (modifiers, requested_codes). modifiers carries {'error': msg} if an
        argument cannot be resolved."""
        mods, requested = {}, set()
        pinmap = {}
        for item in pin or []:
            c, t = self._code(item.get("course")), self._parse_term_label(item.get("term_label"))
            if not c:
                return {"error": f"unknown course '{item.get('course')}'"}, requested
            if t is None:
                return {"error": f"unrecognized term '{item.get('term_label')}'"}, requested
            pinmap[c] = t
            requested.add(c)
        if pinmap:
            mods["pin"] = pinmap

        nb = {}
        for item in delay or []:
            c, t = self._code(item.get("course")), self._parse_term_label(item.get("term_label"))
            if not c:
                return {"error": f"unknown course '{item.get('course')}'"}, requested
            if t is None:
                return {"error": f"unrecognized term '{item.get('term_label')}'"}, requested
            nb[c] = t
            requested.add(c)
        if nb:
            mods["not_before"] = nb

        early = set()
        for course in earliest or []:
            c = self._code(course)
            if not c:
                return {"error": f"unknown course '{course}'"}, requested
            early.add(c)
            requested.add(c)
        if early:
            mods["prefer_early"] = early

        dropset = set()
        for course in drop or []:
            c = self._code(course)
            if not c:
                return {"error": f"unknown course '{course}'"}, requested
            dropset.add(c)
            requested.add(c)
        if dropset:
            mods["drop"] = dropset

        if credit_cap is not None:
            cap = int(credit_cap)
            if cap < 12:
                return {"error": f"credit cap must be at least 12 (got {cap})"}, requested
            mods["credit_cap"] = cap

        base_grad = max(self.base.keys()) if self.base else self.state.current_term
        gb = self._parse_term_label(graduate_by)
        if gb is not None:
            mods["max_grad_term"] = gb
        # "graduate earlier" — express the GOAL as a deadline and let the relaxation
        # search find the enabling overload, rather than guessing a credit cap here.
        if graduate_earlier_by:
            mods["max_grad_term"] = max(base_grad - int(graduate_earlier_by),
                                        self.state.current_term)
        # "graduate later" — relative (model-friendly) or absolute label.
        wants_later = False
        if graduate_later_by:
            mods["min_grad_term"] = base_grad + int(graduate_later_by)
            wants_later = True
        gne = self._parse_term_label(graduate_no_earlier_than)
        if gne is not None:
            mods["min_grad_term"] = gne
            wants_later = True

        # A student volunteering to graduate later almost always wants the extra
        # term to BUY something — a lighter load — not a minimal one-course slip that
        # leaves the peak unchanged. So when they ask to graduate later and don't name
        # an objective, default to spreading the workload. (They can still force a
        # minimal delay by passing objective="minimal_change" explicitly.)
        effective_obj = objective
        if wants_later and not effective_obj:
            effective_obj = "even"

        if effective_obj:
            mods["objective"] = effective_obj
            if effective_obj == "even":
                # 'even' must bound graduation, else the solver spreads courses across
                # the whole horizon to flatten the peak. Use the stated graduation
                # window if given, otherwise hold the current graduation term.
                mods["max_grad_term"] = (mods.get("max_grad_term")
                                         or mods.get("min_grad_term") or base_grad)
        return mods, requested

    def _present(self, cand, requested, credit_cap=None):
        """Build the diff + summary for a feasible candidate and stash it as last_sim."""
        from planner import plan_credits
        d = diff_plans(self.prog, self.state, self.base, cand, requested,
                       credit_cap=credit_cap)
        future = [v for t, v in plan_credits(self.prog, cand).items()
                  if t >= self.state.current_term]
        peak = max(future) if future else 0
        summary = qa._summarize_diff(self.prog, d)
        self.last_sim = {"feasible": True, "candidate": cand, "requested": requested,
                         "credit_cap": credit_cap, "placed": None, "reason": None,
                         "diff": d, "summary": summary, "peak": peak}
        return {"feasible": True, "peak_credits_per_term": peak,
                "graduation_term": term_label(d["grad_term_b"]) if d["grad_term_b"] else None,
                "summary": summary,
                "requested_changes": d["requested_changes"],
                "induced_changes": d["induced_changes"]}

    def compare_plans(self, plan_a=None, plan_b=None):
        """Compare two plans by candidate id (e.g. 'c1' vs 'c2'), or — by default —
        the current plan against the most recent alternative."""
        def resolve(x):
            rec = qa.get_candidate(self.session, x) if x else None
            return rec["plan"] if rec else None
        a = resolve(plan_a) or self.base
        latest = qa.latest_candidate(self.session)
        b = resolve(plan_b) or (latest["plan"] if latest else None)
        if b is None:
            return {"error": "no alternative plan to compare yet; make a change first"}
        d = diff_plans(self.prog, self.state, a, b, set())
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
            # Add a candidate branch the student can hold alongside any others (P3).
            qa.add_candidate(self.session, sim["candidate"],
                             requested=sim["requested"], credit_cap=sim.get("credit_cap"),
                             constraints=sim.get("constraints"), summary=sim["summary"],
                             label=qa._auto_label(self.prog, self.state, sim["candidate"]))
            return _envelope(text or sim["summary"],
                             structured={"kind": "what_if", "summary": sim["summary"]},
                             candidate=qa._plan_view(self.prog, sim["candidate"]),
                             diff=sim["diff"])
        if name == "offer_alternatives":
            # P2+P3: the request was infeasible, but relaxations exist. Add EACH as a
            # holdable candidate branch; show the best one's diff and name the rest.
            sim = self.last_sim or {}
            alts = sim.get("alternatives") or []
            if not alts:
                return self.finalize("report_infeasible", args)
            for a in alts:
                qa.add_candidate(self.session, a["plan"], requested=[],
                                 credit_cap=a["modifiers"].get("credit_cap"),
                                 constraints=self._describe_constraints(a["modifiers"]),
                                 summary=a["description"],
                                 label=f"{a['description']} · {a['graduation_term']}")
            best, others = alts[0], alts[1:]
            d = diff_plans(self.prog, self.state, self.base, best["plan"], set(),
                           credit_cap=best["modifiers"].get("credit_cap"))
            msg = (f"Not possible as asked: {sim.get('reason')} The closest I can do is "
                   f"{best['description']} — graduates {best['graduation_term']} at a peak "
                   f"of {best['peak_credits_per_term']} credits/term (shown).")
            if others:
                msg += " You could also " + "; ".join(
                    f"{a['description']} (graduates {a['graduation_term']})"
                    for a in others) + "."
            return _envelope(text or msg,
                             structured={"kind": "relaxation",
                                         "reason": sim.get("reason"),
                                         "alternatives": [
                                             {k: a[k] for k in ("description",
                                              "graduation_term", "peak_credits_per_term")}
                                             for a in alts]},
                             candidate=qa._plan_view(self.prog, best["plan"]),
                             diff=d)
        if name == "report_infeasible":
            reason = (self.last_sim or {}).get("reason")
            # The new request failed and added nothing — leave any branches the student
            # is already holding untouched (they're still valid options).
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
    prefs_line = ""
    if prefs:
        prefs_line = ("\nKnown student preferences (soft constraints): "
                      + "; ".join(f"{k}={v}" for k, v in prefs.items()) + ".")
    return f"""You are an academic-advising agent for {prog.name}. The student is \
{state.name}; the current term is term {state.current_term} ({term_label(state.current_term)}).{prefs_line}

Answer the student's question by calling tools. Choose each step:
1. OBSERVE — call observe tools to gather verified facts. NEVER state a prerequisite,
   credit count, or eligibility result you did not get from a tool.
   - Use search_course to resolve any course name or partial code to an exact code.
   - Use get_plan_overview to see the full schedule and per-term credit loads.
2. INFER — for ANY plan change, call `replan` ONCE with a constraint set that
   captures the student's intent, then finish. `replan` is the only plan-change
   tool; do not look for a more specific one. Map the request to its arguments
   (combine freely):
   - "take/move X to Y3 Fall", "put X in year 5" → pin: [{{course:X, term_label:"Y3 Fall"}}]
   - "take X earlier / sooner"                   → earliest: ["X"]
   - "delay X / push X back"                     → delay: [{{course:X, term_label: one
                                                    term after its current slot}}]
   - "drop X"                                    → drop: ["X"]
   - "graduate earlier / sooner"                 → graduate_earlier_by: 1 (or N). Do NOT
                                                    set a credit cap; if a heavier load is
                                                    needed the system offers it for you.
   - "take more credits / overwrite the cap"     → credit_cap: the number they gave (or 21)
   - "graduate one term later" / "one year later"→ graduate_later_by: 1 (a term/semester)
                                                    or 2 (a year)
   - "balance / even out / lighten the workload" → objective: "even" (add
                                                    graduate_later_by if they'll accept
                                                    graduating later to achieve it)
   A request can need several at once, e.g. "graduate a year later with an even
   workload" → objective:"even" + graduate_later_by:2. graduate_later_by is relative,
   so you never need to look up the current graduation term to use it. Asking to
   graduate later is almost always a request to LIGHTEN the load over the extra time,
   not just slip one course — so combine graduate_later_by with objective:"even".

   CONTINUITY — if the turn begins with a "[Session context]" note about a pending
   plan or prior constraints, treat a follow-up as a REFINEMENT of that proposal:
   re-issue replan with the same constraints plus the student's tweak (e.g. context
   says objective=even and they now say "make it one semester later" → replan with
   objective:"even" + graduate_later_by:1). Only start fresh if they clearly change
   goals.
3. Finish by calling exactly ONE terminal tool — use this decision rule:
   • Did you call replan AND it returned feasible=true?  → present_plan_change.
   • Did you call replan AND it returned feasible=false?  → report_infeasible.
     (If the result carried "alternatives", they are offered to the student
     automatically — you do not need to do anything extra.)
   • Did you NOT call replan at all?  → respond (lookups/eligibility/audits/recs).
   • Genuinely ambiguous and observing won't resolve it?  → ask_clarification.
     Never ask when the student delegates the decision to you.

Keep replies brief (1-3 sentences), grounded only in tool results. Use plain prose
in sentence case — no markdown, bold, bullets, or headings.

CRITICAL — copy facts from tool results VERBATIM. Repeat term labels exactly as the
tool gives them (e.g. "Y2 Spring"); the only seasons are Fall and Spring (never
"Summer"). Repeat course codes, credit counts, and "need N more" figures exactly as
returned. Do NOT compute term numbers, invent seasons, recalculate or round credits,
or restate a number the tools did not give you. Resolve "this term"/"this semester"/
"now" to term {state.current_term} only when calling tools."""


def _terminal_for_sim(sim):
    """The correct terminal for a completed replan: present the plan if feasible,
    offer relaxations if infeasible-but-recoverable, else report infeasible."""
    if sim.get("feasible"):
        return "present_plan_change"
    if sim.get("alternatives"):
        return "offer_alternatives"
    return "report_infeasible"


def _correct_terminal(name, ctx):
    """Upgrade a wrongly-picked terminal to the right one after a simulation. Small
    models reliably compute the right facts but routinely call respond instead of
    present_plan_change (or report_infeasible when relaxations are on offer). Leaves
    ask_clarification alone. In trust mode (guardrails off) the model's choice
    stands — so its raw reliability is observable."""
    if ctx.last_sim is None or not _GUARDRAILS:
        return name
    if name in ("respond", "present_plan_change", "report_infeasible"):
        corrected = _terminal_for_sim(ctx.last_sim)
        if corrected != name:
            ctx.note(f"corrected_terminal:{name}->{corrected}")
        return corrected
    return name


def _session_briefing(session):
    """Compact, structured cross-turn state that the prose transcript loses: the
    pending candidate branch and the constraints behind it. Injected as a prefix on
    the current turn so the agent has working-memory continuity without replaying raw
    tool calls and without putting dynamic data in the (cacheable) system prompt."""
    cands = session.get("candidates", [])
    last = session.get("last_constraints")
    if not cands and not last:
        return None
    bits = []
    if cands:
        listing = "; ".join(f"{c['id']} = {c.get('label') or 'alternative'}" for c in cands)
        bits.append(f"{len(cands)} alternative plan(s) are pending, not yet accepted: "
                    f"{listing}. A follow-up may refine one, compare two (by id, e.g. "
                    f"compare_plans plan_a=c1 plan_b=c2), accept, or discard.")
    if last:
        bits.append(f"The most recent was produced with these constraints: {last}. If the "
                    f"student refines the SAME goal (e.g. tweaks the timeline of a "
                    f"workload balance), reuse these constraints with the tweak applied; "
                    f"only drop them if they clearly start a new request.")
    return "[Session context] " + " ".join(bits)


def run(prog, state, session, question):
    """Drive the tool loop. Returns the API response envelope. Raises on hard failure
    (qa.handle catches and falls back to the deterministic router). P4: optionally
    attaches an intervention trace so the scaffolding's cost is observable."""
    ctx = _Ctx(prog, state, session)
    env = _agent_loop(ctx, _system_prompt(prog, state, ctx.prefs), session, question)
    if _AGENT_DEBUG:
        env.setdefault("debug", {})["interventions"] = ctx.trace
    return env


def _premature_terminal(name, ctx):
    """True if the model picked a plan-outcome terminal before computing a plan."""
    return name in ("present_plan_change", "offer_alternatives") and ctx.last_sim is None


def _agent_loop(ctx, system, session, question):
    # Replay prior turns (plain user/assistant text) as conversation memory so the
    # agent can resolve references like "take it earlier" or "compare them". The
    # current turn's tool_use/tool_result messages stay local to this loop; only the
    # final answer is persisted to session['history'] (by qa.handle).
    prior = list(session.get("history", []))
    # Prepend a structured working-state briefing to the current turn so pending
    # plan state survives across turns (the transcript only carries prose summaries).
    brief = _session_briefing(session)
    user_content = f"{brief}\n\n{question}" if brief else question
    messages = prior + [{"role": "user", "content": user_content}]

    for _ in range(MAX_STEPS):
        resp = llm_client.chat_tools(system, messages, TOOLS, max_tokens=2048)
        tool_uses = [b for b in resp.content if b.type == "tool_use"]

        if not tool_uses:
            # model answered with plain text and no tool -> treat as a respond()
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return _envelope(text or "Done.")

        # A terminal tool ends the turn. If several tools came back, honour the
        # terminal one and ignore the rest.
        display = next((t for t in tool_uses if t.name in _TERMINAL_TOOLS), None)
        if display is not None:
            # P4: one bounded self-correction. If the model jumps to a plan terminal
            # without having computed a plan (and didn't call replan this turn), nudge
            # it once instead of silently degrading to an empty answer.
            no_infer = not any(t.name in _PLAN_INFER_TOOLS for t in tool_uses)
            if (_GUARDRAILS and no_infer and ctx.corrections_left > 0
                    and _premature_terminal(display.name, ctx)):
                ctx.corrections_left -= 1
                ctx.note(f"self_correct:{display.name}")
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content":
                    f"You called {display.name} but no plan change has been computed. "
                    f"Call replan first, or use respond for a plain answer."})
                continue
            return ctx.finalize(_correct_terminal(display.name, ctx), display.input)

        # Otherwise run the observe/infer tools and feed results back.
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for t in tool_uses:
            out = ctx.dispatch(t.name, t.input)
            if isinstance(out, dict) and out.get("error"):
                ctx.note(f"tool_error:{t.name}")
            results.append({"type": "tool_result", "tool_use_id": t.id,
                            "content": json.dumps(out, ensure_ascii=False)})
            # Auto-finalize after plan-change infer tools so the model cannot
            # second-guess itself into another loop iteration. The terminal is chosen
            # deterministically. In trust mode (guardrails off) we skip this and let
            # the model choose its own terminal next turn, so its reliability shows.
            if (_GUARDRAILS and t.name in _PLAN_INFER_TOOLS
                    and ctx.last_sim is not None):
                messages.append({"role": "user", "content": results})
                term = _terminal_for_sim(ctx.last_sim)
                ctx.note(f"auto_finalize:{term}")
                # Let offer_alternatives build its own rich message; otherwise pass the
                # summary so the model's prose isn't needed.
                args = {} if term == "offer_alternatives" else {
                    "text": ctx.last_sim.get("summary") or out.get("reason") or ""}
                return ctx.finalize(term, args)
        messages.append({"role": "user", "content": results})

    # Ran out of steps without a terminal tool -> force a grounded wrap-up.
    ctx.note("wrapup_forced")
    messages.append({"role": "user",
                     "content": "Now call exactly one terminal tool to finish."})
    resp = llm_client.chat_tools(system, messages, TOOLS, max_tokens=1024,
                                 tool_choice={"type": "any"})
    display = next((b for b in resp.content
                    if b.type == "tool_use" and b.name in _TERMINAL_TOOLS), None)
    if display is not None:
        return ctx.finalize(_correct_terminal(display.name, ctx), display.input)
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return _envelope(text or "I couldn't complete that. Please rephrase.")
