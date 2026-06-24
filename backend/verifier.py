"""
Constraint verification layer (deterministic).

Independent of the planner. Validates an arbitrary multi-term plan against the
VPRS and the student state, returning structured violations. Because it checks
against the spec (not a future outcome), validity and "why-not" are computed with
no student simulator and no LLM.
"""
from vprs import (Program, eval_prereq, offered_in, season_of, describe_prereq,
                  term_label)


class Violation:
    def __init__(self, kind, message, subject=None, term=None, source=None):
        self.kind = kind          # 'prereq' | 'coreq' | 'offering' | 'credit_cap'
                                  # | 'duplicate' | 'category' | 'total' | 'missing'
        self.message = message
        self.subject = subject
        self.term = term
        self.source = source      # audit-trail citation

    def to_dict(self):
        return {"kind": self.kind, "message": self.message,
                "subject": self.subject, "term": self.term, "source": self.source}


def passed_set(state) -> set:
    return {c for c, m in state.passed_marks().items()}


def verify_plan(prog: Program, state, plan: dict, credit_cap: int = None) -> list:
    """
    plan: {term_index: [subject_code, ...]}  (future terms only)
    credit_cap: per-term limit to enforce; defaults to prog.max_credits_per_term.
                Pass a higher value when validating a plan that intentionally
                overloads terms (e.g. "take more credits to graduate earlier").
    Returns list[Violation]. Empty list => valid.
    """
    cap = credit_cap or prog.max_credits_per_term
    v = []
    names = prog.names()
    completed = passed_set(state)         # grows as we walk terms in order
    seen = set(completed)

    for term in sorted(plan.keys()):
        codes = plan[term]
        # duplicates / already passed
        for c in codes:
            if c in completed:
                v.append(Violation("duplicate",
                    f"{c} is already completed but appears in {term_label(term)}.",
                    c, term))
            if codes.count(c) > 1:
                v.append(Violation("duplicate",
                    f"{c} appears more than once in {term_label(term)}.", c, term))

        # per-subject checks use state completed BEFORE this term
        for c in set(codes):
            s = prog.subjects.get(c)
            if not s:
                v.append(Violation("missing", f"Unknown subject {c}.", c, term))
                continue
            if not offered_in(s, term):
                v.append(Violation("offering",
                    f"{c} ({s.name}) is not offered in {season_of(term)} "
                    f"({term_label(term)}); offered: {s.offered}.",
                    c, term, s.source))
            if not eval_prereq(s.prereq, completed):
                v.append(Violation("prereq",
                    f"{c} ({s.name}) is scheduled in {term_label(term)} but its "
                    f"prerequisite is not met: {describe_prereq(s.prereq, names)}.",
                    c, term, s.source))
            # coreqs must be completed earlier or taken this term
            for cor in s.coreq:
                if cor not in completed and cor not in codes:
                    v.append(Violation("coreq",
                        f"{c} ({s.name}) requires corequisite {cor} "
                        f"({names.get(cor, cor)}) in the same or an earlier term.",
                        c, term, s.source))

        # credit cap for the term
        load = sum(prog.subjects[c].credits for c in codes if c in prog.subjects)
        if load > cap:
            v.append(Violation("credit_cap",
                f"{term_label(term)} is overloaded: {load} credits "
                f"(max {cap}).", None, term))

        # commit this term
        for c in codes:
            completed.add(c)
            seen.add(c)

    # graduation-level checks (passed + planned)
    final = completed
    # category minimums
    cat_credits = {}
    for c in final:
        s = prog.subjects.get(c)
        if s:
            cat_credits[s.category] = cat_credits.get(s.category, 0) + s.credits
    for cat, need in prog.category_min_credits.items():
        have = cat_credits.get(cat, 0)
        if have < need:
            v.append(Violation("category",
                f"Category '{cat}' requires {need} credits; plan provides {have}.",
                None, None))
    # total credits
    total = sum(prog.subjects[c].credits for c in final if c in prog.subjects)
    if total < prog.total_credits_required:
        v.append(Violation("total",
            f"Total credits {total} below required {prog.total_credits_required}.",
            None, None))
    # required subjects present
    for c in prog.required_codes:
        if c not in final:
            v.append(Violation("missing",
                f"Required subject {c} ({names.get(c,c)}) is not in the plan.",
                c, None))
    return v


def is_eligible(prog: Program, state, code: str):
    """Single-subject eligibility against current passed set. Returns (bool, reason, source)."""
    s = prog.subjects.get(code)
    if not s:
        return False, f"Unknown subject {code}.", None
    completed = passed_set(state)
    if code in completed:
        return False, f"{code} is already completed.", None
    if not eval_prereq(s.prereq, completed):
        return (False,
                f"prerequisite not met: {describe_prereq(s.prereq, prog.names())}",
                s.source)
    return True, "all prerequisites satisfied", s.source
