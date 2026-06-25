# System Architecture

## Overview and Design Principle

ATLAS is built around a single architectural commitment: **the language
model orchestrates but never adjudicates.** Every claim that reaches the
student — a prerequisite, a credit count, an eligibility verdict, a
graduation date, the feasibility of a requested change — is computed by a
deterministic, independently checkable core. The model's role is confined to
two things it is actually good at: (i) interpreting an under-specified natural
language request and mapping it onto a structured call against that core, and
(ii) rendering the core's verified output as fluent prose. It contributes no
domain facts of its own. This separation is what lets the system use a small,
even locally-hosted, model without inheriting that model's tendency to
confabulate requirements or miscount credits.

The system is organized as five layers, from the specification of the
curriculum up to the user-facing service:

```
 ┌──────────────────────────────────────────────────────────────┐
 │  Presentation / Service     app.py (FastAPI) · index.html     │
 ├──────────────────────────────────────────────────────────────┤
 │  Orchestration              agent.py                          │
 │   policy ⟨S, A, π, ρ⟩, tool space, scaffolding, memory        │
 ├───────────────┬──────────────────────────────────────────────┤
 │ Session/State │ Model Client    llm_client.py                 │
 │  qa.py        │  provider abstraction, parse/render fallback  │
 ├───────────────┴──────────────────────────────────────────────┤
 │  Deterministic Reasoning Core                                 │
 │   planner.py (CP-SAT / greedy) · verifier.py · diff.py        │
 ├──────────────────────────────────────────────────────────────┤
 │  Specification (VPRS)       vprs.py · student_data.py         │
 │   Subject, Program, prerequisite expression trees, provenance │
 └──────────────────────────────────────────────────────────────┘
```

The remainder of this section describes each layer bottom-up, then returns to
the orchestration layer to formalize the agent as a policy and to describe the
deterministic scaffolding that makes a weak model usable.

## The Specification Layer: A Verifiable Program Requirement Specification

The foundation of the system is the **Verifiable Program Requirement
Specification (VPRS)**, a declarative encoding of the degree program
(`vprs.py`). The central design decision here is that prerequisites are *not*
modeled as plain dependency edges. Real catalogs state requirements as logical
formulae — "A and one of {B, C}", "at least two of {…}" — and flattening these
to edges loses exactly the structure an advisor must reason over. VPRS
therefore represents each prerequisite as an **expression tree** over four
node types:

- an **atom** `{"subject": "CS101"}`,
- **AND** / **OR** of sub-expressions, and
- **k-of-n** `{"op": "KOF", "k": 2, "args": [...]}`,

with `None` denoting no prerequisite. A single recursive evaluator,
`eval_prereq(expr, completed)`, decides satisfaction against a set of completed
courses; companion functions extract the atom set (for dependency analysis)
and render a human-readable description.

Each `Subject` carries its code, name, credit value, requirement category,
offering pattern (`fall` / `spring` / `both`), prerequisite and corequisite
expressions, and — critically — a **`source` string**: the catalog or course-
outline sentence the requirement was extracted from. This provenance field is
the audit trail that makes the specification re-checkable and allows the system
to answer a "why can't I take X?" question by citing a real regulation rather
than a generated justification. The `Program` aggregates subjects, the set of
required codes, per-category minimum-credit requirements, a per-term credit
cap, and the planning horizon. The student record (`student_data.py`) supplies
the immutable history — courses passed and the current term — against which all
reasoning is performed.

## The Deterministic Reasoning Core

Three modules turn the specification into answers. All of them are pure,
deterministic Python; none of them call a language model.

### Planning (`planner.py`)

Plan construction is the heart of the core, exposed through a single interface,
`plan_path(prog, state, modifiers, anchor)`, that returns a uniform
`(plan, infeasible_reason)` pair. Two interchangeable backends sit behind it:

- A **CP-SAT model** (Google OR-Tools), used by default. It is *complete*: it
  returns a graduation-feasible plan when one exists and **proves
  infeasibility** when none does. The model places a boolean `x[c,t]` for each
  schedulable course `c` and future term `t`. Required courses are constrained
  to exactly one term and electives to at most one; prerequisite expression
  trees are compiled into **reified booleans over "completed before term t"**
  (AND → min, OR → max, k-of-n → a cardinality constraint), corequisites into
  "completed by end of term t", and offerings, credit caps, and category-credit
  minimums into linear constraints. Category minimums are what compel the
  solver to *choose* electives, so plans satisfy the degree, not merely the
  required list.
- A **greedy backend** — prerequisite-gated, credit-capped, topological in
  spirit — used only when OR-Tools is unavailable or explicitly selected. It is
  fast but incomplete, and a CP *infeasible* result is treated as authoritative
  (the greedy planner is never consulted to "second-guess" it).

The same call site supports a set of **what-if modifiers** that compose freely:
pin a course to a term, forbid a course before a term, drop a course, prefer
scheduling a course early, override the global per-term credit cap, set
per-term credit caps for specific terms, and bound graduation from above
(`max_grad_term`) or below (`min_grad_term`). The lower bound deserves note: to
force a genuinely later graduation it is not enough to bound the
graduation-term variable (a free variable the objective would simply minimize
back down); the model instead *requires at least one course at or after the
bound*, which is what actually pushes the schedule out.

Over these constraints the planner exposes a **parameterized objective** —
a single knob with three settings, all sharing the same constraint set:

- **`early`** — minimize the graduation term, then honor "take earlier"
  preferences, then front-load. This is the default for a fresh plan.
- **`even`** — minimize the *peak* per-term credit load to spread workload
  evenly, with graduation bounded so the solver balances within a window rather
  than smearing courses across the whole horizon.
- **`minimal_change`** — when re-solving from an `anchor` (the current plan),
  minimize deviation from that baseline so a single request produces a diff
  showing only the requested change and its genuine forced consequences, not a
  wholesale re-selection of electives.

Finally, `relax(prog, state, base, modifiers)` turns an infeasible request into
options. Given a constraint set the solver could not satisfy, it walks a small,
prioritized ladder of **single-constraint relaxations** — raise the per-term
cap, allow graduating a little later than asked, drop an impossible pin — and
returns those that become feasible, best (least-compromising) first. The ladder
is **goal-preserving**: for an over-tight *earlier*-graduation request it offers
only heavier terms, never the self-defeating "graduate later." This is the
mechanism behind the system's ability to negotiate ("can't do X as asked, but
raising the cap to 21 would work") instead of dead-ending.

### Verification and Differencing (`verifier.py`, `diff.py`)

`verifier.py` independently checks eligibility (`is_eligible`) and validates a
whole plan (`verify_plan`) against the specification — prerequisites,
corequisites, offerings, and credit caps — returning both a verdict and the
source-cited reason. Because verification is separate from planning, a plan can
be checked by a path that did not produce it; this is also what the evaluation
harness uses as its oracle. `diff.py` computes a structured `diff_plans` between
two plans: the requested changes, the *induced* downstream changes a request
forced, the graduation-term delta, and a validity flag for the alternative.
The diff is the unit the frontend renders and the unit the orchestrator
summarizes.

## The Model Client and Graceful Degradation (`llm_client.py`)

The model client abstracts over two transports behind one interface: Claude via
the Anthropic SDK (default; a small model such as Haiku) and any
OpenAI-compatible local endpoint such as LM Studio, selected by the
`LLM_PROVIDER` environment variable. It exposes three capabilities: a tool-use
turn (`chat_tools`, used by the agent), and a `parse`/`render` pair used by the
deterministic fallback path. To keep transports interchangeable, the client
converts Anthropic-style tool schemas and message lists to and from the OpenAI
function-calling format, wrapping local responses in lightweight objects so the
orchestrator is agnostic to which backend served a turn.

Two properties make this layer robust. First, **availability is cached** with a
short TTL, so a request never blocks probing an absent model. Second, and more
importantly, **every model-dependent path degrades gracefully**: if the backend
is unreachable or returns malformed output, `parse` falls back to a
keyword-and-pattern parser and `render` falls back to deterministic templates.
The application therefore remains fully usable — answering lookups, eligibility,
audits, and basic what-ifs — with no model at all. This fallback router is not
merely a safety net; it is also the **model-free baseline (B0)** in our
evaluation.

## The Session and Routing Layer (`qa.py`)

`qa.handle(prog, state, session, question)` is the single entry point for a
turn. It records conversation history centrally (so both the agentic and
fallback paths contribute to it), then dispatches: if a tool-capable model is
available, it invokes the agent; otherwise — or if the agent raises — it runs
the deterministic parse-then-route path. Both paths share the *same*
deterministic resolvers, so the facts a student receives are identical
regardless of which path served the turn; only the orchestration and phrasing
differ.

This layer also owns the **multi-candidate working state**. Rather than a single
"last alternative" slot that each new what-if overwrites, the session holds up
to `MAX_CANDIDATES` alternative plan branches, each with an id, a label, the
constraints that produced it, and a cached summary. A student can hold several
proposals at once, compare any two by id, and accept or discard them
individually. Capping at one branch (`ATLAS_MAX_CANDIDATES=1`) recovers the
overwrite behavior and is used as a substitution ablation in the evaluation.

## The Orchestration Layer (`agent.py`)

The orchestrator is where the language model does its work, and its design is
the paper's primary contribution. Three commitments shape it.

### A unified, composable action

Earlier iterations of such systems route each phrasing to a bespoke endpoint
("delay course", "rebalance", "graduate later"), which makes the system brittle
at exactly the compositional requests advising produces. ATLAS instead exposes a
**single plan-change capability, `replan`**, whose arguments are the composable
modifiers of the planner: `pin`, `earliest`, `delay`, `drop`, `credit_cap`,
per-term `term_caps`, `graduate_earlier_by` / `graduate_later_by` /
`graduate_by`, and the `objective` knob. Any combination may be supplied in one
call — pin a course *and* raise the cap *and* graduate later — and the model's
job is to translate intent into that argument set rather than to select among
many narrow tools. The remaining tools fall into three roles: **observe**
(gather verified facts — requirements, eligibility, audit, recommendation,
search, plan overview), **infer** (`replan`, `compare_plans`), and **display**
(the terminal tools that finalize the reply: `respond`, `present_plan_change`,
`report_infeasible`, `ask_clarification`). The display tool the model chooses
determines the API response shape.

### A three-layer memory model

The per-turn execution context (`_Ctx`) exposes memory at three scopes and
enforces what may be written:

- **Structural** (read-only): the program specification and the student's
  immutable record.
- **Episodic** (read/write): per-session working state — the current plan, the
  held candidate branches, and the constraints behind the most recent
  proposal. Tools may write *only* this layer.
- **Long-term** (read view): student preferences as soft constraints, seeded
  into the session.

Cross-turn continuity is carried two ways. Prior user/assistant turns are
replayed as conversation memory so the model can resolve references such as
"take *it* earlier" or "compare *them*." Separately — because a prose transcript
loses structured state — a compact **session briefing** is prepended to the
current turn, naming the pending candidate branches (by id) and the constraints
that produced the latest one, so a follow-up is treated as a *refinement* of an
existing proposal rather than a fresh request. Suppressing this briefing
(`ATLAS_BRIEFING=off`), leaving only the raw transcript, is the memory
substitution ablation.

### Deterministic scaffolding, made observable and switchable

A small local model reliably computes the right *facts* (because the tools do
that) but is unreliable about *control flow*: it will call `respond` after a
successful `replan` instead of `present_plan_change`, or jump to a plan-outcome
terminal before computing a plan. ATLAS surrounds the loop with deterministic
guardrails that compensate: **auto-finalization** after a plan-change infer tool
(the terminal is chosen deterministically from the simulation result, so the
model cannot loop back and mis-finalize), **terminal correction** (upgrading a
wrongly chosen display tool to the correct one), and **one bounded
self-correction** nudging the model when it terminates prematurely. Every
intervention is recorded in a trace.

Crucially, this scaffolding is **load-bearing but measurable**. Each guardrail
is gated by an environment flag so it can be switched off and its contribution
quantified (`ATLAS_GUARDRAILS=off` for a "trust mode" that lets a stronger
model's raw control-flow reliability show; `ATLAS_AGENT_DEBUG=1` to attach the
intervention trace to the response). The relaxation search and multi-candidate
store are likewise toggleable. We stress that these flags implement
**substitution ablations, not deletions**: turning a capability off does not
remove it from the system, it *hands that capability back to the model* — with
relaxation off, an infeasible result is passed to the model to react to itself,
rather than being handled by the engineered negotiate step. This lets the
evaluation ask the sharp question — does the *engineered policy* beat the model
doing the same job ad hoc? — rather than the trivial one.

### The orchestrator as a policy ⟨S, A, π, ρ⟩

These pieces compose into an explicit orchestration policy, which we state
formally so that it, rather than the grounding, is the object of study:

- **State `S_t`** = (the replayed dialogue history, the resolved constraint set
  Γ_t for the current goal, the set of held candidate branches C_t, and the
  program/student specification P).
- **Action space `A`** = {clarify, observe, solve, augment, negotiate, report,
  compare, respond} — the policy-level actions realized by the tool calls and
  terminal choice at each turn (e.g. *negotiate* is realized when an infeasible
  `replan` is answered with relaxations; *augment* when a re-plan refines a
  branch the student was already holding).
- **Policy `π`** = the agent loop together with its deterministic scaffolding:
  given `S_t`, choose tool calls and a terminal action, subject to
  auto-finalization and terminal correction.
- **Metric `ρ`** = per-turn set-valued action validity (was the action taken in
  the set of acceptable actions for that turn?) plus end-of-dialogue task
  success, both scored by the deterministic core as oracle.

This formalization is the bridge to the evaluation: the conditions studied are
substitutions over `π` (briefing, relaxation, multi-candidate) and a free-form
baseline that keeps the same `A` but removes the engineered orchestration,
measured by `ρ` on multi-turn and negotiation dialogues.

## The Service and Presentation Layer

A small FastAPI application (`app.py`) exposes the system: `GET /api/state`
returns the program, the student record, the current plan, and the held
candidate branches; `POST /api/chat` runs a turn through `qa.handle`; and
`/api/reschedule`, `/api/accept`, and `/api/discard` support direct
manipulation of plans and branches (drag-and-drop a course to a term, adopt a
branch as the plan, or drop a branch by id). Session state is held in memory for
a single demo user; a production deployment would substitute per-user storage
behind the same interface. The single-page frontend (`index.html`) renders the
current plan and each pending candidate as a separate lane with per-branch
accept/compare/discard controls, making the multi-candidate model directly
visible to the student.

## Summary

The architecture's through-line is the strict separation between a
**verifiable deterministic core** — a logic-bearing requirement specification, a
complete CP-SAT planner with a parameterized objective and a goal-preserving
relaxation search, and independent verification and differencing — and a
**language-model orchestrator** that interprets requests, composes a single
unified plan-change action, maintains structured multi-turn working state, and
narrates verified results. The deterministic scaffolding that makes a small
model viable is not hidden but exposed as switchable, measurable policy
components, so the contribution under study — the orchestration policy itself —
can be ablated and evaluated against the model handling the same responsibilities
on its own.
