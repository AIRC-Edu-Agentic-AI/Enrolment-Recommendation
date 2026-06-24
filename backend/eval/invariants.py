"""
Deterministic scorer for the advisor benchmark.

The system is its own oracle: every plan and fact is computed by the deterministic
core, so a request's *correct* outcome is whatever the planner produces from the gold
constraints. Scoring therefore reduces to checking invariants on the produced state —
no human judgement at eval time.

A `Result` bundles everything one item's run produced; `score_item` returns whether
the terminal and every outcome invariant held (pass/fail), plus a separate
translation score (did the agent pick the right `replan` args — RQ2).
"""
from dataclasses import dataclass, field

from vprs import term_label
from planner import plan_credits, grad_term
import agent

_parse_term = agent._Ctx._parse_term_label

# Deterministic-router answer kinds -> the observe tool the agent path would call,
# so A-type "called" invariants score under both the agent and the fallback.
_KIND_TO_TOOL = {
    "prereq": "lookup_requirements",
    "eligibility": "check_eligibility",
    "why_not": "check_eligibility",
    "graduation_audit": "graduation_audit",
    "recommend": "recommend_courses",
}

_OBSERVE_TOOLS = {"lookup_requirements", "check_eligibility", "graduation_audit",
                  "recommend_courses", "search_course", "get_plan_overview"}


def derive_action(terminal, tools, had_prior_candidate):
    """Map a turn's realized (terminal, tools) onto the policy action space A
    (see POLICY.md). `augment` vs `solve` is distinguished by whether the student was
    already holding a candidate when this turn re-planned (i.e. a cross-turn refine)."""
    tools = set(tools or [])
    if terminal == "ask_clarification":
        return "clarify"
    if terminal == "offer_alternatives":
        return "negotiate"
    if terminal == "report_infeasible":
        return "report"
    if "compare_plans" in tools:
        return "compare"
    if terminal == "present_plan_change":
        return "augment" if had_prior_candidate else "solve"
    if tools & _OBSERVE_TOOLS:
        return "observe"
    return "respond"


@dataclass
class Result:
    env: dict                       # last turn's response envelope
    session: dict                   # session state after all turns
    baseline_plan: dict             # the plan before any change (current_plan)
    tool_calls: list = field(default_factory=list)   # aggregated across turns
    turns: list = field(default_factory=list)        # per-turn {terminal, tools, prior_cands}
    fell_back: bool = False         # agent raised and the deterministic router answered

    def actions(self):
        """The policy action realized at each turn (POLICY.md action space)."""
        return [derive_action(t["terminal"], t["tools"], t["prior_cands"] > 0)
                for t in self.turns]

    # --- derived views -----------------------------------------------------
    def candidate(self):
        cands = self.session.get("candidates") or []
        return cands[-1]["plan"] if cands else None

    def candidates(self):
        return [c["plan"] for c in (self.session.get("candidates") or [])]

    def terminal(self):
        d = self.env.get("debug") or {}
        if d.get("terminal"):
            return d["terminal"]
        # Derive from the envelope shape (deterministic router has no debug).
        kind = (self.env.get("structured") or {}).get("kind")
        if self.env.get("infeasible"):
            return "report_infeasible"
        if kind == "relaxation":
            return "offer_alternatives"
        if kind == "what_if" and self.env.get("candidate"):
            return "present_plan_change"
        if kind == "clarify" or self.env.get("needs_input"):
            return "ask_clarification"
        return "respond"

    def called_tools(self):
        names = {c["name"] for c in self.tool_calls}
        kind = (self.env.get("structured") or {}).get("kind")
        if kind in _KIND_TO_TOOL:           # fallback path: infer the tool from kind
            names.add(_KIND_TO_TOOL[kind])
        return names

    def replan_input(self):
        for c in self.tool_calls:
            if c["name"] == "replan":
                return c.get("input") or {}
        return None


# --- outcome invariants -----------------------------------------------------
def _peak(prog, plan, t0):
    return max((v for t, v in plan_credits(prog, plan).items() if t >= t0), default=0)


def _check(prog, t0, inv, r):
    k = inv["kind"]
    cand = r.candidate()
    if k == "no_candidate":
        return not r.candidates()
    if k == "course_in_term":
        return cand is not None and inv["course"] in cand.get(_parse_term(inv["term_label"]), [])
    if k == "course_not_in_term":
        return cand is None or inv["course"] not in cand.get(_parse_term(inv["term_label"]), [])
    if k == "course_present":
        return cand is not None and any(inv["course"] in cs for cs in cand.values())
    if k == "course_absent":
        return cand is not None and all(inv["course"] not in cs for cs in cand.values())
    if k == "term_cap":
        return cand is not None and plan_credits(prog, cand).get(_parse_term(inv["term_label"]), 0) <= inv["max"]
    if k == "all_terms_cap":
        return cand is not None and all(v <= inv["max"] for v in plan_credits(prog, cand).values())
    if k == "grad_delta":
        return cand is not None and (grad_term(cand) - grad_term(r.baseline_plan)) == inv["delta"]
    if k == "peak_below_baseline":
        return cand is not None and _peak(prog, cand, t0) < _peak(prog, r.baseline_plan, t0)
    if k == "relaxation_offered":
        if r.terminal() != "offer_alternatives" or not r.candidates():
            return False
        if "grad_delta" in inv:
            bg = grad_term(r.baseline_plan)
            return any(grad_term(p) - bg == inv["grad_delta"] for p in r.candidates())
        return True
    if k == "called":
        return inv["tool"] in r.called_tools()
    raise ValueError(f"unknown invariant kind: {k}")


def _check_translation(inv, replan_in):
    """Did the agent's replan call carry the expected argument? (RQ2; agent path only)."""
    if replan_in is None:
        return False
    arg = inv["arg"]
    val = replan_in.get(arg)
    if val is None:
        return False
    if "value" in inv:
        return val == inv["value"]
    if "course" in inv:                      # list-of-codes or list-of-objects
        codes = [x.get("course") if isinstance(x, dict) else x for x in val]
        ok = inv["course"] in codes
        if "term_label" in inv:
            tgt = _parse_term(inv["term_label"])
            ok = ok and any(isinstance(x, dict) and x.get("course") == inv["course"]
                            and _parse_term(x.get("term_label")) == tgt for x in val)
        return ok
    if "term_label" in inv:                  # term_caps: [{term_label, max_credits}]
        tgt = _parse_term(inv["term_label"])
        return any(_parse_term(x.get("term_label")) == tgt
                   and x.get("max_credits") == inv.get("max_credits") for x in val)
    return True


def score_item(prog, state, item, r):
    gold = item["gold"]
    t0 = state.current_term
    term = r.terminal()
    if "terminal_any" in gold:
        terminal_ok = term in gold["terminal_any"]
    else:
        terminal_ok = term == gold.get("terminal")

    inv_results = [(inv, _check(prog, t0, inv, r)) for inv in gold.get("invariants", [])]
    outcome_ok = all(ok for _, ok in inv_results)

    trans = gold.get("translation", [])
    replan_in = r.replan_input()
    trans_results = [(inv, _check_translation(inv, replan_in)) for inv in trans]
    translation_ok = all(ok for _, ok in trans_results) if trans else None

    # Action validity (RQ1 metric): is the action chosen at each turn in the gold
    # acceptable-action set? Sets are set-valued (several actions can be correct).
    turn_gold = gold.get("turn_actions")
    actions = r.actions()
    action_valid = action_total = 0
    bad_actions = []
    if turn_gold:
        for i, acceptable in enumerate(turn_gold):
            if not acceptable:                       # unconstrained turn -> skip
                continue
            action_total += 1
            chosen = actions[i] if i < len(actions) else None
            if chosen in set(acceptable):
                action_valid += 1
            else:
                bad_actions.append({"turn": i, "chose": chosen, "expected": acceptable})
    action_validity = (action_valid / action_total) if action_total else None

    return {
        "id": item["id"], "type": item["type"], "tier": item["tier"],
        "terminal": term, "terminal_ok": terminal_ok,
        "outcome_ok": outcome_ok,
        "task_success": terminal_ok and outcome_ok,
        "passed": terminal_ok and outcome_ok,          # alias (back-compat)
        "translation_ok": translation_ok,
        "actions": actions,
        "action_validity": action_validity,
        "action_perfect": (action_validity == 1.0) if action_validity is not None else None,
        "bad_actions": bad_actions,
        "fell_back": r.fell_back,
        "failed_invariants": [inv["kind"] for inv, ok in inv_results if not ok],
    }
