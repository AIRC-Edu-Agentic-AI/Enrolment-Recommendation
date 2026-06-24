"""
Benchmark runner for the ATLAS advising policy.

Drives each labeled dialogue through the system under several conditions and scores
it with the deterministic invariant checker. Runs IN-PROCESS so it can reset state
per item and read the full session + per-turn debug trace.

The contribution under study is the orchestration POLICY (see eval/POLICY.md), so the
conditions are SUBSTITUTION ablations — each hands one capability back to the model
rather than deleting it — plus a free-form baseline:

  B0     deterministic fallback router (model-free)
  FULL   full engineered policy (= G2)
  FF     free-form: same tools/action space, NO engineered orchestration
  -mem   FULL minus structured cross-turn state (raw transcript only)
  -relax FULL minus the negotiate action (model reacts to infeasibility itself)
  -mc    FULL minus held alternatives (single overwrite slot)

Usage:
  cd backend
  python -m eval.run_eval                    # all available conditions, 1 rep
  python -m eval.run_eval --reps 5
  python -m eval.run_eval --conditions B0 FF FULL
"""
import argparse
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vprs import PROGRAM
from student_data import demo_state
from planner import plan_path
import qa
import agent
import llm_client
from eval.invariants import Result, score_item, _KIND_TO_TOOL

HERE = os.path.dirname(os.path.abspath(__file__))

# condition -> (guardrails, briefing, relax, max_candidates); B0 handled specially.
CONDITIONS = {
    "FULL":  (True,  True,  True,  3),
    "FF":    (False, False, False, 1),
    "-mem":  (True,  False, True,  3),
    "-relax": (True, True,  False, 3),
    "-mc":   (True,  True,  True,  1),
}


def _fresh():
    state = demo_state()
    base, _ = plan_path(PROGRAM, state)
    session = {"current_plan": base, "candidates": [], "history": []}
    return state, base, session


def _turn_terminal(env):
    d = env.get("debug") or {}
    if d.get("terminal"):
        return d["terminal"]
    kind = (env.get("structured") or {}).get("kind")
    if env.get("infeasible"):
        return "report_infeasible"
    if kind == "relaxation":
        return "offer_alternatives"
    if kind == "what_if" and env.get("candidate"):
        return "present_plan_change"
    if kind == "clarify" or env.get("needs_input"):
        return "ask_clarification"
    return "respond"


def _turn_tools(env):
    d = env.get("debug") or {}
    tools = {c["name"] for c in d.get("tool_calls", [])}
    kind = (env.get("structured") or {}).get("kind")
    if kind in _KIND_TO_TOOL:                     # deterministic router: infer tool
        tools.add(_KIND_TO_TOOL[kind])
    return tools


def _run_item(item, condition):
    state, base, session = _fresh()
    agent._AGENT_DEBUG = True
    saved_offline = None
    if condition == "B0":
        # model-free baseline: force the rule-based parser + templates
        saved_offline = (llm_client.llm_available, llm_client.agent_available)
        llm_client.llm_available = lambda *a, **k: False
        llm_client.agent_available = lambda *a, **k: False
    else:
        g, b, rx, mc = CONDITIONS[condition]
        agent._GUARDRAILS, agent._BRIEFING, agent._RELAX = g, b, rx
        qa.MAX_CANDIDATES = mc

    turns, tool_calls, env, fell_back = [], [], {}, False
    try:
        for turn in item["turns"]:
            prior_cands = len(session.get("candidates", []))
            env = qa.handle(PROGRAM, state, session, turn)
            fell_back = fell_back or env.get("parse_source") == "fallback"
            tools = _turn_tools(env)
            tool_calls += (env.get("debug") or {}).get("tool_calls", [])
            turns.append({"terminal": _turn_terminal(env), "tools": tools,
                          "prior_cands": prior_cands})
    finally:
        if saved_offline:
            llm_client.llm_available, llm_client.agent_available = saved_offline
        qa.MAX_CANDIDATES = 3
        agent._GUARDRAILS, agent._BRIEFING, agent._RELAX = True, True, True

    r = Result(env=env, session=session, baseline_plan=base, tool_calls=tool_calls,
               turns=turns, fell_back=fell_back)
    return score_item(PROGRAM, state, item, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--types", nargs="+", default=None,
                    help="keep only items whose type startswith any of these (e.g. C D)")
    ap.add_argument("--benchmark", default=os.path.join(HERE, "benchmark.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.benchmark, encoding="utf-8") if l.strip()]
    if args.types:
        items = [it for it in items if any(it["type"].startswith(p) for p in args.types)]
    model_ok = llm_client.agent_available()
    if args.conditions:
        conditions = args.conditions
    else:
        conditions = ["B0"] + (["FULL", "FF", "-mem", "-relax", "-mc"] if model_ok else [])
    if not model_ok and any(c != "B0" for c in conditions):
        print("! No tool-capable model reachable — model conditions skipped.\n")
        conditions = [c for c in conditions if c == "B0"] or ["B0"]

    rows, out = [], open(args.out, "w", encoding="utf-8")
    for cond in conditions:
        reps = 1 if cond == "B0" else args.reps
        for item in items:
            for rep in range(reps):
                res = _run_item(item, cond)
                res["condition"], res["rep"] = cond, rep
                rows.append(res)
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
    out.close()
    _report(rows, conditions)
    print(f"\nWrote {len(rows)} rows to {args.out}")


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


def _fmt(x):
    return f"{x:>8.0%}" if x is not None else f"{'—':>8}"


def _report(rows, conditions):
    D = lambda r: r["type"].startswith("D")
    C = lambda r: r["type"].startswith("C")

    print("=" * 70)
    print("RQ1 - Orchestration policy on the turns that need it")
    print("                     multi-turn (D)          negotiation (C)")
    print("  condition        act-valid   task        act-valid   task")
    for c in conditions:
        cd = [r for r in rows if r["condition"] == c and D(r)]
        cc = [r for r in rows if r["condition"] == c and C(r)]
        print(f"  {c:<14}" + _fmt(_avg(cd, "action_validity")) + _fmt(_avg(cd, "task_success"))
              + "   " + _fmt(_avg(cc, "action_validity")) + _fmt(_avg(cc, "task_success")))

    print("\nTable 3 - Component ablation (task success by slice)")
    print("  variant          overall    comp   multi(D)  negot(C)")
    tier = lambda r, t: r["tier"] == t
    for c in conditions:
        cr = [r for r in rows if r["condition"] == c]
        print(f"  {c:<14}" + _fmt(_avg(cr, "task_success"))
              + _fmt(_avg([r for r in cr if tier(r, "compositional")], "task_success"))
              + _fmt(_avg([r for r in cr if D(r)], "task_success"))
              + _fmt(_avg([r for r in cr if C(r)], "task_success")))

    print("\nOverall (all items)")
    print("  condition        task   action  transl  fellback")
    for c in conditions:
        cr = [r for r in rows if r["condition"] == c]
        fb = sum(1 for r in cr if r["fell_back"])
        print(f"  {c:<14}" + _fmt(_avg(cr, "task_success")) + _fmt(_avg(cr, "action_validity"))
              + _fmt(_avg(cr, "translation_ok")) + f"{fb:>9}")


if __name__ == "__main__":
    main()
