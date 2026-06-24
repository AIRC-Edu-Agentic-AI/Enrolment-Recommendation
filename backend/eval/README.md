# Advisor evaluation harness

Measures how reliably the agent turns natural-language advising requests into the
**correct plan change + terminal action** — and how much the deterministic
scaffolding contributes. The system is its own oracle: every plan is computed by the
deterministic core, so a request's correct outcome is whatever the planner produces
from the gold constraints, and scoring is fully automatic (no human-in-the-loop).

## Files
- `benchmark.jsonl` — labeled utterances on a **type × tier** grid (see below).
- `invariants.py` — the deterministic scorer (terminal + outcome invariants, plus a
  separate translation/RQ2 score).
- `run_eval.py` — in-process driver over the conditions; prints aggregates, writes
  `results.jsonl`.

## Conditions (the ablation ladder)
| | What | Toggles |
|---|---|---|
| **B0** | deterministic fallback router (model-free) | LLM forced off |
| **B1** | agent, **trust mode** (raw model reliability) | `ATLAS_GUARDRAILS=off` |
| **B2** | agent, full shipped system | `ATLAS_GUARDRAILS=on` |

`B0 < B1 < B2` quantifies what the agent and the scaffolding each add; **B2(weak
model) vs B1(strong model)** is the P4 question — can a stronger model in trust mode
match a weak one propped up by scaffolding?

## Question types (capability axis) × tiers (difficulty axis)
**Types:** A informational (lookup/eligibility/why-not/audit/recommend) · B plan
change (pin/earlier/delay/drop/global-cap/term-cap/graduate-later/balance) ·
C negotiation (graduate-earlier→relaxation / hard-infeasible / impossible-pin) ·
D dialog (compare / cross-turn refine) · E negative (out-of-scope / ambiguous /
unknown-course). **Tiers:** single · compositional · adversarial (paraphrase, typo,
Vietnamese).

## Metrics
- **pass** = terminal correct **and** all outcome invariants hold (primary; works in
  every condition).
- **translation** = did the agent's `replan` call carry the right arguments? (RQ2;
  agent conditions only — isolates the LLM's mapping from the solver.)
- **fell_back** = agent raised and the deterministic router answered (a failure mode).
- Pass rates are broken out by tier and type.

## Run
```bash
cd backend
python -m eval.run_eval --conditions B0          # model-free baseline (fast)
python -m eval.run_eval --reps 3                 # all available conditions
```
B1/B2 need a tool-capable model reachable (Anthropic key, or LM Studio running a
tool-calling model such as Qwen). To compare models, run with different `.env`
settings and tag the output. Agent conditions are stochastic — use `--reps ≥ 3`.

## Benchmark item schema
```json
{
  "id": "B14-single-termcap", "type": "B14", "tier": "single", "lang": "en",
  "turns": ["I can only do 15 credits in Y2 Spring"],
  "gold": {
    "terminal": "present_plan_change",
    "invariants": [{"kind": "term_cap", "term_label": "Y2 Spring", "max": 15}],
    "translation": [{"arg": "term_caps", "term_label": "Y2 Spring", "max_credits": 15}]
  }
}
```
`turns` is a list (multi-turn items run in sequence on one session). Use
`terminal_any: [...]` when more than one terminal is acceptable. Invariant kinds:
`course_in_term`, `course_not_in_term`, `course_present`, `course_absent`,
`term_cap`, `all_terms_cap`, `grad_delta`, `peak_below_baseline`,
`relaxation_offered`, `no_candidate`, `called`.

## Scope / honest limits
- This measures **translation + orchestration**, not planner correctness — gold
  outcomes come from the same solver the agent uses. Planner correctness is covered
  separately by `backend/test_replan.py`.
- The seed benchmark (~30 items) is a runnable skeleton; a real study wants ~150,
  weighted toward B/C/D where the agent earns its keep.
- Negative types (E) use a proxy invariant (no fabricated candidate); they do not
  semantically verify the prose refusal.
