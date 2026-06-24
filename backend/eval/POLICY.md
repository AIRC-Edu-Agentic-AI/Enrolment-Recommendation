# The advising policy as the object of study

The contribution is an **orchestration policy** for multi-turn, constraint-grounded
advising — not the solver (a solver satisfying constraints is true by construction)
and not fact-grounding (grounding removing hallucination is true by definition). The
policy is what decides, *across turns*, which action to take given an evolving,
partially-specified goal. This file specifies it as ⟨S, A, π, ρ⟩ so the ablations
measure a defined object rather than "the system."

## State  S_t
The information the policy acts on at turn *t*:

| Component | Meaning | In code |
|---|---|---|
| `h_t` | dialogue so far (utterances + prior answers) | `session["history"]` |
| `Γ_t` | what has been **resolved** about the student's goal & constraints | `session["last_constraints"]` + the briefing's carried intent |
| `C_t` | alternative plans currently **held** | `session["candidates"]` |
| `P`   | the accepted/current plan being edited | `session["current_plan"]` |

`Γ_t` and `C_t` are *engineered* state: the system maintains them explicitly and
serializes them into each turn via `_session_briefing` (the explicit rendering of
S_t). The substitution ablations below remove that engineering, not the information.

## Action space  A
The moves the policy may select (each grounded — facts/feasibility come from
deterministic tools, never the model):

| Action | Meaning | Realization |
|---|---|---|
| `clarify` | ask a question when the goal is under-specified | `ask_clarification` |
| `observe` | fetch a verified fact (prereq, eligibility, audit, recommend, search, plan) | observe tools → `respond` |
| `solve(Γ)` | re-plan under the resolved constraints/objective | `replan(...)` → `present_plan_change` |
| `augment(Γ, +c)` | re-solve with a constraint **added to the carried Γ** (cross-turn refine) | `replan(...)` reusing prior `Γ_t` |
| `negotiate` | on infeasibility, report it **and propose a minimal relaxation** | `replan` infeasible → `offer_alternatives` |
| `report` | infeasibility with no reasonable relaxation | `report_infeasible` |
| `compare(c_i, c_j)` | contrast two held plans | `compare_plans` |
| `respond` | give the grounded answer / present | terminal text |

## Policy  π : S_t → A
π is realized by the model under the system prompt **plus** the deterministic
scaffolding (auto-finalize, terminal correction, one self-correction — see P4). π is
what the paper claims; the prompt and guardrails are its parameters.

## Metric  ρ  (two levels)
1. **Action validity** (per turn): is the action π selected a member of the gold
   **acceptable-action set** `A*_t`? Acceptable sets are *set-valued* because more than
   one action is often correct (e.g. `clarify` *or* `solve` under a sensible default).
   Report precision/recall of selected vs acceptable actions over turns.
2. **Task success** (end of dialogue): does the final state satisfy the resolved goal?
   (the outcome invariants in `invariants.py`).

The agent's realized action per turn is derived from `debug.terminal` + `tool_calls`
(see `derive_action` in `invariants.py`).

## Conditions = ablations of π (substitution, not deletion)
Each ablation hands one capability back to the model so we test whether the
*engineered* policy beats the model doing it ad hoc — the question whose answer is
**not** known a priori.

| Condition | Removes from π | Flag | The model still has… |
|---|---|---|---|
| **Full** | nothing | (defaults) | — |
| **−memory** | structured `Γ_t`/`C_t` briefing | `ATLAS_BRIEFING=off` | the raw transcript `h_t` |
| **−relaxation** | the `negotiate` action | `ATLAS_RELAX=off` | the infeasibility reason, free to react |
| **−multi-candidate** | holding `C_t` (overwrite) | `ATLAS_MAX_CANDIDATES=1` | one candidate at a time |
| **−scaffolding** (trust) | auto-finalize + terminal correction | `ATLAS_GUARDRAILS=off` | full free choice of action |
| **Free-form (FF)** | all engineering | all of the above | same tools, no policy |

**FF is the load-bearing baseline:** same action space and same grounded tools, no
engineered orchestration. If Full beats FF on the turns that need cross-turn state and
negotiation (tiers C/D), the policy is the contribution. If a strong model makes FF
competitive, the finding localizes *when* structure helps (the model × policy
interaction) — still a finding, not a strawman.

## Why this ordering
RQ1 (the policy on C/D — multi-turn + negotiation) is where outcomes are genuinely
uncertain. RQ2 (the G0→G1→G2 grounding ladder) is supporting evidence that the facts
and feasibility the policy relies on are themselves trustworthy — it is not the
contribution, because its result is close to a priori.
