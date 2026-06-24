"""
QA router. Parse -> route to a deterministic resolver -> structured answer ->
grounded render. What-if questions produce a candidate plan + a structured diff,
which the frontend renders as a branch.
"""
import os

from vprs import describe_prereq, term_label, season_of
from verifier import is_eligible, verify_plan
from planner import plan_path, plan_credits, grad_term
from diff import diff_plans
import llm_client


# How many prior user/assistant turns to replay as conversation memory. Kept small
# so each agent turn stays cheap; the agent only needs recent context for anaphora
# ("take it earlier", "compare them").
MAX_HISTORY_TURNS = 6


def handle(prog, state, session, question):
    """
    session: mutable dict holding 'current_plan', 'candidates' (held alternative
    branches), 'last_constraints', and 'history' (conversation memory).
    Returns a response dict for the API.

    When a tool-capable backend is available, an agent decides which tools to call
    (observe -> infer -> display). Otherwise we fall back to the deterministic
    parse-then-route path below. The agent and the fallback share the same
    deterministic resolvers, so facts are identical either way.

    Conversation history is recorded here (centrally) so both paths contribute to it
    and the agent path can resolve references to earlier turns.
    """
    history = session.setdefault("history", [])
    if llm_client.agent_available():
        try:
            import agent
            resp = agent.run(prog, state, session, question)
        except Exception:
            resp = handle_deterministic(prog, state, session, question)
    else:
        resp = handle_deterministic(prog, state, session, question)
    _record_turn(history, question, resp.get("text", ""))
    return resp


def _record_turn(history, question, answer):
    """Append this turn to conversation memory and trim to the last few turns."""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer or "(no answer)"})
    keep = MAX_HISTORY_TURNS * 2
    if len(history) > keep:
        del history[:len(history) - keep]


def handle_deterministic(prog, state, session, question):
    """Parse the question into a typed intent, then route to a resolver."""
    idx = prog.names()
    q = llm_client.parse_question(question, idx, state.current_term)
    intent = q["intent"]
    subjects = [c for c in q.get("subjects", []) if c in prog.subjects]

    if intent == "regulatory_lookup":
        answer = _resolve_lookup(prog, subjects)
    elif intent == "eligibility":
        answer = _resolve_eligibility(prog, state, subjects)
    elif intent == "why_not":
        answer = _resolve_why_not(prog, state, subjects)
    elif intent == "recommend":
        answer = _resolve_recommend(prog, state, session)
    elif intent == "graduation_audit":
        answer = _resolve_audit(prog, state)
    elif intent == "what_if":
        return _resolve_what_if(prog, state, session, q, question)
    else:
        answer = {"kind": "out_of_scope"}

    return {"text": llm_client.render(answer), "structured": answer,
            "intent": intent, "parse_source": q.get("_source"),
            "candidate": None, "diff": None}


# --- resolvers --------------------------------------------------------------
def _resolve_lookup(prog, subjects):
    if not subjects:
        return {"kind": "out_of_scope"}
    c = subjects[0]
    s = prog.subjects[c]
    return {"kind": "prereq", "subject": c, "subject_name": s.name,
            "prereq_text": describe_prereq(s.prereq, prog.names()),
            "credits": s.credits, "category": s.category,
            "offered": s.offered, "source": s.source}


def _resolve_eligibility(prog, state, subjects):
    if not subjects:
        return {"kind": "out_of_scope"}
    c = subjects[0]
    ok, reason, src = is_eligible(prog, state, c)
    return {"kind": "eligibility", "subject": c,
            "subject_name": prog.subjects[c].name,
            "eligible": ok, "reason": reason, "source": src}


def _resolve_why_not(prog, state, subjects):
    if not subjects:
        return {"kind": "out_of_scope"}
    c = subjects[0]
    ok, reason, src = is_eligible(prog, state, c)
    if ok:
        return {"kind": "eligibility", "subject": c,
                "subject_name": prog.subjects[c].name,
                "eligible": True, "reason": reason, "source": src}
    return {"kind": "why_not", "subject": c,
            "subject_name": prog.subjects[c].name, "reason": reason,
            "source": src or "program catalog"}


def _resolve_recommend(prog, state, session):
    plan = session.get("current_plan")
    if not plan:
        plan, _ = plan_path(prog, state)
        session["current_plan"] = plan
    t = state.current_term
    codes = plan.get(t, [])
    return {"kind": "recommend", "term": t, "term_label": term_label(t),
            "subjects": [(c, prog.subjects[c].name) for c in codes],
            "credits": sum(prog.subjects[c].credits for c in codes)}


def _resolve_audit(prog, state):
    passed = set(state.passed_marks().keys())
    remaining = [(c, prog.subjects[c].name)
                 for c in prog.required_codes if c not in passed]
    cat = {}
    for c in passed:
        s = prog.subjects.get(c)
        if s:
            cat[s.category] = cat.get(s.category, 0) + s.credits
    short = [f"{k} (need {need - cat.get(k,0)} more)"
             for k, need in prog.category_min_credits.items()
             if cat.get(k, 0) < need]
    return {"kind": "graduation_audit", "remaining": remaining,
            "categories_short": ", ".join(short)}


# --- what-if (produces candidate + diff) ------------------------------------
_ACTION_TO_MOD = {
    # delay: push the named course to no earlier than the term after its current slot
    "delay": "not_before",
    "prefer_early": "prefer_early",
    "drop": "drop",
    "pin": "pin",
}


def _resolve_what_if(prog, state, session, q, question):
    base = session.get("current_plan")
    if not base:
        base, _ = plan_path(prog, state)
        session["current_plan"] = base

    action = q.get("action")
    subjects = [c for c in q.get("subjects", []) if c in prog.subjects]
    # "I want this course this term" often parses as what_if with a named subject but
    # no explicit action -> treat it as pinning that course into the current term.
    if not action and subjects:
        action = {"type": "pin", "code": subjects[0], "term": state.current_term}

    # no action and no subject -> compare the latest pending candidate to current
    if not action:
        rec = latest_candidate(session)
        if not rec:
            ans = {"kind": "what_if",
                   "summary": "There's no alternative plan to compare yet. "
                              "Tell me a change to try, e.g. 'take Database Systems this "
                              "semester', 'delay Operating Systems', or 'take Machine "
                              "Learning earlier'."}
            return {"text": llm_client.render(ans), "structured": ans,
                    "intent": "what_if", "parse_source": q.get("_source"),
                    "candidate": None, "diff": None}
        d = diff_plans(prog, state, base, rec["plan"],
                       set(rec.get("requested") or []), credit_cap=rec.get("credit_cap"))
        ans = {"kind": "what_if", "summary": _summarize_diff(prog, d)}
        return {"text": llm_client.render(ans), "structured": ans,
                "intent": "what_if", "parse_source": q.get("_source"),
                "candidate": _plan_view(prog, rec["plan"]), "diff": d}

    sim = simulate_change(prog, state, base, action)
    if not sim["feasible"]:  # infeasible -> HITL, no fake branch
        ans = {"kind": "infeasible", "reason": sim["reason"]}
        return {"text": llm_client.render(ans), "structured": ans,
                "intent": "what_if", "parse_source": q.get("_source"),
                "candidate": None, "diff": None, "infeasible": True}

    add_candidate(session, sim["candidate"], requested=sim["requested"],
                  summary=sim["summary"],
                  label=_auto_label(prog, state, sim["candidate"]))
    ans = {"kind": "what_if", "summary": sim["summary"]}
    return {"text": llm_client.render(ans), "structured": ans,
            "intent": "what_if", "parse_source": q.get("_source"),
            "candidate": _plan_view(prog, sim["candidate"]), "diff": sim["diff"]}


def simulate_change(prog, state, base, action):
    """Compute a what-if without touching session state. Shared by the deterministic
    router and the agent's `simulate_plan_change` tool.

    action: {"type": delay|prefer_early|drop|pin, "code": str, "term": int|null}
    Returns a dict:
      {"feasible": bool, "reason": str|None, "candidate": plan|None,
       "diff": dict|None, "requested": set, "summary": str, "placed": int|None}
    """
    code = action.get("code")
    typ = action.get("type")
    requested = {code} if code else set()
    modifiers = {}
    pin_term = None

    if typ == "delay" and code:
        cur = next((t for t, cs in base.items() if code in cs), None)
        target = (action.get("term") or (cur + 2 if cur else state.current_term + 1))
        modifiers = {"not_before": {code: target}}
    elif typ == "prefer_early" and code:
        modifiers = {"prefer_early": {code}}
    elif typ == "drop" and code:
        modifiers = {"drop": {code}}
    elif typ == "pin" and code:
        # "take X this term" leaves term null -> default to the current term.
        # not_before stops the greedy scheduler from sliding it to an earlier term,
        # so the course actually lands in the requested term.
        pin_term = action.get("term") or state.current_term
        modifiers = {"pin": {code: pin_term}, "not_before": {code: pin_term}}

    # Anchor the re-solve to the baseline so the diff shows only the requested
    # change and its forced consequences, not a fresh elective selection.
    cand, reason = plan_path(prog, state, modifiers=modifiers, anchor=base)
    if reason:
        return {"feasible": False, "reason": reason, "candidate": None,
                "diff": None, "requested": requested, "summary": reason,
                "placed": None}

    d = diff_plans(prog, state, base, cand, requested)
    placed = next((t for t, cs in cand.items() if code in cs), None) if code else None
    if typ == "pin" and code:
        where = term_label(placed) if placed else "your plan"
        summary = (f"{prog.subjects[code].name} ({code}) is scheduled in "
                   f"{where}. " + _summarize_diff(prog, d))
    else:
        summary = _summarize_diff(prog, d)
    return {"feasible": True, "reason": None, "candidate": cand, "diff": d,
            "requested": requested, "summary": summary, "placed": placed}


def _summarize_diff(prog, d):
    parts = []
    gd = d.get("grad_delta")
    if gd is None:
        parts.append("graduation term unchanged")
    elif gd == 0:
        parts.append("you still graduate in " + term_label(d["grad_term_b"]))
    elif gd > 0:
        parts.append(f"graduation moves {gd} term(s) later to " + term_label(d["grad_term_b"]))
    else:
        parts.append(f"graduation moves {-gd} term(s) earlier to " + term_label(d["grad_term_b"]))
    rq = d["requested_changes"]
    ind = d["induced_changes"]
    if rq:
        parts.append(f"{len(rq)} requested change(s)")
    if ind:
        parts.append(f"{len(ind)} forced downstream change(s)")
    if not d["valid_b"]:
        parts.append("but the alternative is NOT valid")
    return "; ".join(parts) + "."


def _plan_view(prog, plan):
    """Serialize a plan for the frontend."""
    cr = plan_credits(prog, plan)
    return {
        "terms": [
            {"term": t, "label": term_label(t),
             "subjects": [{"code": c, "name": prog.subjects[c].name,
                           "credits": prog.subjects[c].credits,
                           "category": prog.subjects[c].category}
                          for c in plan[t]],
             "credits": cr[t]}
            for t in sorted(plan.keys())],
        "grad_term": grad_term(plan),
    }


# --- multi-candidate working state (P3) -------------------------------------
# The session holds up to MAX_CANDIDATES alternative branches the student can hold,
# compare, and accept individually — instead of a single overwrite-on-each-change slot.
# ATLAS_MAX_CANDIDATES=1 is the multi-candidate SUBSTITUTION ablation (overwrite).
MAX_CANDIDATES = int(os.environ.get("ATLAS_MAX_CANDIDATES", "3"))


def _new_candidate_id(session):
    n = session.get("_cand_seq", 0) + 1
    session["_cand_seq"] = n
    return f"c{n}"


def add_candidate(session, plan, requested=None, credit_cap=None,
                  constraints=None, label=None, summary=None):
    """Append a candidate branch (FIFO-capped at MAX_CANDIDATES). Returns its id.
    Also tracks the most-recent constraints for the cross-turn briefing (P1)."""
    cands = session.setdefault("candidates", [])
    cid = _new_candidate_id(session)
    cands.append({"id": cid, "plan": plan, "requested": list(requested or []),
                  "credit_cap": credit_cap, "constraints": constraints,
                  "label": label, "summary": summary})
    if len(cands) > MAX_CANDIDATES:
        del cands[0]
    if constraints:
        session["last_constraints"] = constraints
    return cid


def get_candidate(session, cid):
    return next((r for r in session.get("candidates", []) if r["id"] == cid), None)


def latest_candidate(session):
    cands = session.get("candidates", [])
    return cands[-1] if cands else None


def clear_candidates(session):
    session["candidates"] = []
    session["last_constraints"] = None


def _auto_label(prog, state, plan):
    g = grad_term(plan)
    cr = plan_credits(prog, plan)
    peak = max((v for t, v in cr.items() if t >= state.current_term), default=0)
    return f"Graduate {term_label(g)} · peak {peak} cr" if g else "Alternative"


def candidate_view(prog, state, base, rec):
    """Serialize one candidate for the API: plan view + diff vs the current plan."""
    plan = rec["plan"]
    d = diff_plans(prog, state, base, plan, set(rec.get("requested") or []),
                   credit_cap=rec.get("credit_cap"))
    cr = plan_credits(prog, plan)
    peak = max((v for t, v in cr.items() if t >= state.current_term), default=0)
    return {"id": rec["id"],
            "label": rec.get("label") or _auto_label(prog, state, plan),
            "summary": rec.get("summary"),
            "plan": _plan_view(prog, plan), "diff": d,
            "grad_term": grad_term(plan), "peak": peak}
