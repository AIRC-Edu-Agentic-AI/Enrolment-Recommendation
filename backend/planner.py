"""
Registration planning agent (deterministic core).

Two interchangeable backends behind one interface (`plan_path`):
  - CP-SAT (OR-Tools): a complete constraint model — finds a graduation-feasible
    plan if one exists, and proves infeasibility when none does. Default.
  - Greedy: topological-style, prerequisite-gated, credit-capped. Fast but
    incomplete (can miss a feasible plan). Used as a fallback when OR-Tools is
    unavailable, or when PLANNER=greedy.

Both honor the same "what-if" modifiers (pin a subject to a term, forbid a subject
before a term, drop a subject, prefer scheduling a subject early) and return the
same `(plan, infeasible_reason)` shape, so the chat layer is unaffected by which
backend runs.
"""
import os

from vprs import Program, eval_prereq, offered_in, prereq_atoms, season_of

try:
    from ortools.sat.python import cp_model
    _CP_OK = True
except Exception:  # OR-Tools not installed -> greedy only
    _CP_OK = False


def plan_path(prog: Program, state, modifiers=None, anchor=None):
    """Plan the remaining terms. Returns (plan, infeasible_reason).

    `anchor` (a baseline plan {term: [codes]}) makes the CP backend prefer the
    smallest change from that baseline — used for what-if re-solves so a single
    request doesn't trigger a wholesale elective re-pick. Ignored by the greedy
    backend.

    Dispatches to the CP-SAT backend when available (and not disabled via
    PLANNER=greedy); falls back to greedy only if OR-Tools is missing or the
    CP model raises (a CP *infeasible* result is authoritative and is returned
    as-is — greedy is not consulted, since it could miss a real solution)."""
    if _CP_OK and os.environ.get("PLANNER", "cp").strip().lower() != "greedy":
        try:
            return _plan_cp(prog, state, modifiers, anchor)
        except Exception:
            return _plan_greedy(prog, state, modifiers)
    return _plan_greedy(prog, state, modifiers)


def _blocking_score(prog, code, pool):
    """How many pool subjects (transitively) depend on `code`. Higher = schedule earlier."""
    score = 0
    for other in pool:
        if other == code:
            continue
        if code in prereq_atoms(prog.subjects[other].prereq):
            score += 1
    return score


def _plan_greedy(prog: Program, state, modifiers=None):
    """
    Greedy backend. Returns (plan, infeasible_reason).
      plan: {term_index: [codes]} for terms >= state.current_term, or {} if infeasible.
      modifiers: dict with optional keys:
        - pin:   {code: term_index}     force code into a specific term
        - drop:  set(codes)             remove from target set
        - prefer_early: set(codes)      bump priority (e.g. user asked to take sooner)
        - not_before: {code: term}      forbid scheduling code before given term
    """
    modifiers = modifiers or {}
    pin = modifiers.get("pin", {})
    drop = set(modifiers.get("drop", set()))
    prefer_early = set(modifiers.get("prefer_early", set()))
    not_before = modifiers.get("not_before", {})

    passed = set(state.passed_marks().keys())
    target = [c for c in prog.required_codes if c not in passed and c not in drop]

    # carry any pinned subjects even if not in required set
    for c in pin:
        if c not in target and c not in passed:
            target.append(c)

    completed = set(passed)
    to_schedule = list(target)
    plan = {}

    for term in range(state.current_term, prog.horizon + 1):
        season = season_of(term)
        cap = prog.max_credits_per_term
        load = 0
        chosen = []

        # 1) honor pins for this term first
        pinned_here = [c for c in to_schedule if pin.get(c) == term]
        # 2) eligible pool
        def eligible(c):
            s = prog.subjects[c]
            if not offered_in(s, term):
                return False
            if c in not_before and term < not_before[c]:
                return False
            return eval_prereq(s.prereq, completed)

        pool = [c for c in to_schedule if c not in pinned_here and eligible(c)]
        # priority: pinned, then prefer_early, then most-blocking, then fewer credits
        pool.sort(key=lambda c: (
            0 if c in prefer_early else 1,
            -_blocking_score(prog, c, to_schedule),
            prog.subjects[c].credits))
        ordered = pinned_here + pool

        for c in ordered:
            s = prog.subjects[c]
            # re-check eligibility for pinned (a pin can be infeasible)
            if c not in pinned_here and not eligible(c):
                continue
            if c in pinned_here and not offered_in(s, term):
                # pin into a term where the course isn't offered -> infeasible request
                return {}, (f"Cannot place {c} in term {term} ({season_of(term)}): "
                            f"it is only offered in {s.offered}.")
            if c in pinned_here and not eval_prereq(s.prereq, completed):
                # pin violates prereqs at this term -> infeasible request
                return {}, (f"Cannot place {c} in term {term}: its prerequisite "
                            f"is not satisfied by then.")
            if load + s.credits > cap:
                continue
            # coreq must be completed or also chosen this term
            if any(cor not in completed and cor not in chosen for cor in s.coreq):
                # try to pull coreq in if eligible & fits
                pulled = False
                for cor in s.coreq:
                    cs = prog.subjects.get(cor)
                    if (cs and cor in to_schedule and offered_in(cs, term)
                            and eval_prereq(cs.prereq, completed)
                            and load + s.credits + cs.credits <= cap):
                        chosen.append(cor); load += cs.credits; pulled = True
                if not pulled and any(cor not in completed for cor in s.coreq):
                    continue
            chosen.append(c)
            load += s.credits

        if chosen:
            plan[term] = chosen
            for c in chosen:
                completed.add(c)
                if c in to_schedule:
                    to_schedule.remove(c)

    if to_schedule:
        return {}, ("No feasible path: could not schedule "
                    + ", ".join(to_schedule) + " within "
                    + f"{prog.horizon} terms under the constraints.")
    return plan, None


# ---------------------------------------------------------------------------
# CP-SAT backend (OR-Tools)
# ---------------------------------------------------------------------------
def _plan_cp(prog: Program, state, modifiers=None, anchor=None):
    """Constraint-model backend. Same contract as _plan_greedy, but complete:
    it returns a graduation-feasible plan when one exists and a reason when none
    does. Encodes prerequisites (AND/OR/k-of-n) as reified booleans over
    "completed before this term", plus coreqs, offerings, credit caps, and the
    what-if modifiers. Objective: graduate as early as possible, then honor
    prefer_early, then front-load.
    """
    modifiers = modifiers or {}
    pin = modifiers.get("pin", {})
    drop = set(modifiers.get("drop", set()))
    prefer_early = set(modifiers.get("prefer_early", set()))
    not_before = modifiers.get("not_before", {})

    passed = set(state.passed_marks().keys())
    t0, T = state.current_term, prog.horizon
    terms = list(range(t0, T + 1))

    # Compulsory courses we MUST place (required, minus passed/dropped).
    must = [c for c in prog.required_codes if c not in passed and c not in drop]
    # Elective pool the planner MAY place to satisfy category credit minimums:
    # every other catalog subject not passed/dropped/required.
    req_set = set(prog.required_codes)
    electives = [c for c in prog.subjects
                 if c not in req_set and c not in passed and c not in drop]
    # A pinned course must be taken even if it's an elective.
    for c in pin:
        if c not in passed and c not in drop and c not in must and c not in electives:
            electives.append(c)
    sched = must + electives          # everything the model may schedule
    sched_set = set(sched)
    if not sched:
        return {}, None

    # Specific, user-facing pre-checks for pins (clearer than a bare INFEASIBLE).
    for c, t in pin.items():
        if c in passed or c in drop:
            continue
        s = prog.subjects[c]
        if t < t0:
            return {}, (f"Cannot place {c} in term {t}: that term has already passed "
                        f"(current term is {t0}).")
        if t > T:
            return {}, (f"Cannot place {c} in term {t}: only {T} terms are planned.")
        if not offered_in(s, t):
            return {}, (f"Cannot place {c} in term {t} ({season_of(t)}): "
                        f"it is only offered in {s.offered}.")

    # Electives the user explicitly named (pin / "take earlier") must be taken, not
    # merely allowed — otherwise the optimizer may drop them from the plan.
    force_take = set(pin) | set(prefer_early)

    model = cp_model.CpModel()
    x = {(c, t): model.NewBoolVar(f"x_{c}_{t}") for c in sched for t in terms}

    for c in sched:
        if c in must or c in force_take:
            model.AddExactlyOne(x[c, t] for t in terms)       # required / requested
        else:
            model.AddAtMostOne(x[c, t] for t in terms)        # elective: optional
        s = prog.subjects[c]
        for t in terms:
            if (not offered_in(s, t)) or (c in not_before and t < not_before[c]):
                model.Add(x[c, t] == 0)
        if c in pin:
            model.Add(x[c, pin[c]] == 1)

    for t in terms:  # credit cap
        model.Add(sum(prog.subjects[c].credits * x[c, t] for c in sched)
                  <= prog.max_credits_per_term)

    # Category credit minimums: passed credits + scheduled credits in each category
    # must reach the requirement. This is what makes the planner CHOOSE electives.
    for cat, need in prog.category_min_credits.items():
        passed_cr = sum(prog.subjects[c].credits for c in passed
                        if c in prog.subjects and prog.subjects[c].category == cat)
        planned = sum(prog.subjects[c].credits * x[c, t]
                      for c in sched if prog.subjects[c].category == cat for t in terms)
        if passed_cr < need:
            model.Add(planned >= need - passed_cr)

    # "completed before t" / "completed by end of t" as 0/1 linear expressions
    def done_before(code, t):
        if code in passed:
            return 1
        if code not in sched_set:
            return 0
        return sum(x[code, tt] for tt in terms if tt < t)

    def done_through(code, t):
        if code in passed:
            return 1
        if code not in sched_set:
            return 0
        return sum(x[code, tt] for tt in terms if tt <= t)

    def as_bool(expr):
        if isinstance(expr, int):
            b = model.NewBoolVar("")
            model.Add(b == expr)
            return b
        b = model.NewBoolVar("")
        model.Add(expr >= 1).OnlyEnforceIf(b)
        model.Add(expr == 0).OnlyEnforceIf(b.Not())
        return b

    def prereq_bool(expr, t):
        """Reified boolean: is `expr` satisfied by courses completed before term t?"""
        if not expr:
            return as_bool(1)
        if "subject" in expr:
            return as_bool(done_before(expr["subject"], t))
        kids = [prereq_bool(a, t) for a in expr["args"]]
        res = model.NewBoolVar("")
        op = expr["op"]
        if op == "AND":
            model.AddMinEquality(res, kids)
        elif op == "OR":
            model.AddMaxEquality(res, kids)
        elif op == "KOF":
            k = expr["k"]
            model.Add(sum(kids) >= k).OnlyEnforceIf(res)
            model.Add(sum(kids) <= k - 1).OnlyEnforceIf(res.Not())
        else:
            raise ValueError(f"unknown op {op}")
        return res

    for c in sched:
        s = prog.subjects[c]
        for t in terms:
            if s.prereq:  # taking c at t requires its prereqs met before t
                model.AddImplication(x[c, t], prereq_bool(s.prereq, t))
            for cor in s.coreq:  # coreq completed by end of t (same term or earlier)
                model.AddImplication(x[c, t], as_bool(done_through(cor, t)))

    last = model.NewIntVar(t0, T, "last")
    for c in sched:
        for t in terms:
            model.Add(last >= t).OnlyEnforceIf(x[c, t])
    earliness = sum(t * x[c, t] for c in sched for t in terms)
    early_pref = sum(t * x[c, t] for c in sched if c in prefer_early for t in terms)

    if anchor:
        # What-if re-solve: keep as close to the baseline as possible so the diff
        # shows only the requested change and its genuine forced consequences (not a
        # fresh elective selection). deviation = (baseline courses that left their
        # slot or were dropped) + (non-baseline courses newly added).
        anchor_term = {c: t for t, codes in anchor.items() for c in codes
                       if c in sched_set}
        # Courses the request is about are governed by their modifier, not anchored
        # (otherwise the deviation penalty would cancel the requested move/add).
        requested_codes = set(pin) | prefer_early | set(not_before) | drop
        dev = []
        for c in sched:
            if c in requested_codes:
                continue
            if c in anchor_term and (c, anchor_term[c]) in x:
                dev.append(1 - x[c, anchor_term[c]])      # penalize moving/dropping
            elif c not in anchor_term:
                dev.append(sum(x[c, t] for t in terms))   # penalize adding
        deviation = sum(dev)
        # Priority: (1) graduate as early as feasible, (2) minimal change from the
        # baseline, (3) front-load. So a delay that fits keeps graduation on-time with
        # the fewest shifts; a delay that doesn't fit pushes graduation later instead
        # of churning the whole plan. The requested course is excluded from deviation
        # above, so its own placement follows its modifier.
        model.Minimize(100000 * last + 1000 * deviation + earliness)
    else:
        # Fresh plan: minimize graduation term, then prefer_early, then front-load.
        # Minimizing earliness also discourages taking electives beyond the minimum.
        model.Minimize(10000 * last + 100 * early_pref + earliness)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if pin:
            c = next(iter(pin))
            return {}, (f"Cannot place {c} in term {pin[c]}: its prerequisites "
                        f"cannot be satisfied by then under the program constraints.")
        return {}, (f"No feasible path: the remaining requirements do not fit in "
                    f"terms {t0}-{T} under the credit and prerequisite constraints.")

    plan = {}
    for t in terms:
        codes = sorted(c for c in sched if solver.Value(x[c, t]) == 1)
        if codes:
            plan[t] = codes
    return plan, None


def plan_credits(prog, plan):
    return {t: sum(prog.subjects[c].credits for c in codes)
            for t, codes in plan.items()}


def grad_term(plan):
    return max(plan.keys()) if plan else None
