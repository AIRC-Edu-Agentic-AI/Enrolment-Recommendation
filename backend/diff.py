"""
Plan-diff resolver (deterministic).

Compares two validated plans and produces a structured comparison object. The key
move: changes are split into REQUESTED (subjects the student explicitly named) and
INDUCED (forced downstream consequences). Both the chat answer and the
agent-volunteered comparison render from this one object; the LLM never compares
plans freehand.
"""
from vprs import Program
from planner import plan_credits, grad_term
from verifier import verify_plan


def _flatten(plan):
    """code -> term."""
    return {c: t for t, codes in plan.items() for c in codes}


def diff_plans(prog: Program, state, plan_a: dict, plan_b: dict, requested: set):
    """plan_a = baseline (current), plan_b = candidate. requested = explicitly named codes."""
    fa, fb = _flatten(plan_a), _flatten(plan_b)
    a_codes, b_codes = set(fa), set(fb)

    def tag(c):
        return "requested" if c in requested else "induced"

    added = [{"code": c, "name": prog.subjects[c].name, "term": fb[c],
              "tag": tag(c)} for c in sorted(b_codes - a_codes)]
    removed = [{"code": c, "name": prog.subjects[c].name, "term": fa[c],
                "tag": tag(c)} for c in sorted(a_codes - b_codes)]
    moved = [{"code": c, "name": prog.subjects[c].name,
              "from": fa[c], "to": fb[c], "tag": tag(c)}
             for c in sorted(a_codes & b_codes) if fa[c] != fb[c]]

    # earliest term where the two plans differ -> fork point
    fork = None
    all_terms = sorted(set(plan_a) | set(plan_b))
    for t in all_terms:
        if set(plan_a.get(t, [])) != set(plan_b.get(t, [])):
            fork = t
            break

    ca, cb = plan_credits(prog, plan_a), plan_credits(prog, plan_b)
    load = []
    for t in all_terms:
        load.append({"term": t, "a": ca.get(t, 0), "b": cb.get(t, 0)})

    va = [x.to_dict() for x in verify_plan(prog, state, plan_a)]
    vb = [x.to_dict() for x in verify_plan(prog, state, plan_b)]

    ga, gb = grad_term(plan_a), grad_term(plan_b)
    grad_delta = (gb - ga) if (ga and gb) else None

    return {
        "fork_term": fork,
        "grad_term_a": ga, "grad_term_b": gb, "grad_delta": grad_delta,
        "added": added, "removed": removed, "moved": moved,
        "load_by_term": load,
        "valid_a": len(va) == 0, "valid_b": len(vb) == 0,
        "violations_a": va, "violations_b": vb,
        "requested_changes": [x for x in (added + moved + removed)
                              if x["tag"] == "requested"],
        "induced_changes": [x for x in (added + moved + removed)
                            if x["tag"] == "induced"],
    }
