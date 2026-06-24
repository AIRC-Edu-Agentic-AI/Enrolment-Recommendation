# ATLAS Course Advisor (pilot)

A course-registration advising app for a bachelor program. The student sees their
enrollment history and current inferred plan; they chat to ask questions or request
a change, and the app shows each alternative as a **branch** off the current plan —
several can be held at once — with **requested** vs **forced (induced)** changes
colored differently and a structured comparison.

The LLM is a **tool-using agent**, not a knowledge source. For each question it
**observes** verified facts (prerequisites, eligibility, audit, course search, plan
overview) and, for any plan change, calls a single composable **`replan`** tool that
maps the request to a constraint set — pin/move a course, take one earlier, delay one,
drop one, raise the per-term credit cap, graduate earlier or later, or balance the
workload — which can be combined in one call. Every regulatory and planning fact —
prerequisites, credit math, what's feasible, why something is blocked — is computed by
deterministic Python (OR-Tools CP-SAT) against a logic-bearing requirement spec; the
tools wrap that backend, so the model orchestrates but never invents facts.

When a request is **infeasible**, the agent doesn't dead-end: it searches for the
minimal, goal-preserving **relaxation** that would make it possible (e.g. "graduating a
term earlier needs 21 credits/term — here's that plan") and offers it as an acceptable
branch. Alternatives accumulate as **named candidate branches** the student can hold,
compare, and accept individually, and the agent carries pending branches and their
constraints across turns so a follow-up *refines* the last idea instead of restarting.

If a tool-capable model is offline (or `LLM_PROVIDER=lmstudio` with a model that lacks
tool calling), the app falls back to a rule-based parser + template renderer and stays
fully usable.

The backend is configurable via `.env` (see `.env.example`). By default the project
runs on **Claude Haiku** through the Anthropic API; set `LLM_PROVIDER=lmstudio` to use
a local OpenAI-compatible model (e.g. LM Studio) instead.

---

## Run it

### 1. Configure the model
```bash
cd atlas-advisor
cp .env.example .env
```
Then edit `.env` and set `ANTHROPIC_API_KEY` (get one at
https://console.anthropic.com). The default `LLM_PROVIDER=anthropic` runs on
**Claude Haiku** (`ANTHROPIC_MODEL=claude-haiku-4-5`).

To use a local model instead, set `LLM_PROVIDER=lmstudio`, then start LM Studio:
download an instruct model (e.g. Llama-3.1-8B-Instruct or Qwen2.5-7B/8B-Instruct
GGUF), load it, open the **Local Server** tab, **Start Server** (default endpoint
`http://localhost:1234/v1`), and leave it running.

> **Note (local models):** the agent path needs a model that supports **tool
> calling**. Instruct models like Qwen2.5/Qwen3 and Llama-3.1 work; pure reasoning
> models (e.g. DeepSeek-R1) do not and will route to the deterministic fallback.

### 2. Start the app
```bash
pip install -r requirements.txt
cd backend
uvicorn app:app --reload --port 8000
```
Open http://localhost:8000

The header shows a green dot when it can reach the configured model, red (fallback)
otherwise. Each chat reply shows `parsed by: llm` or `parsed by: fallback` so you can
see which path handled it.

### 3. Run the tests
```bash
cd backend
python -m unittest test_replan      # 48 deterministic tests, no model/network needed
```
These cover constraint composition, the relaxation search, cross-turn memory, the
multi-candidate state, and the agent's guardrails (driven by a scripted fake LLM).

### Configuration
All settings live in `.env` (read by `backend/llm_client.py` via python-dotenv):
```bash
LLM_PROVIDER=anthropic               # "anthropic" (default) or "lmstudio"
ANTHROPIC_API_KEY=sk-ant-...         # required for the anthropic provider
ANTHROPIC_MODEL=claude-haiku-4-5     # model to call
# Local-model alternative (must support tool calling):
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
LMSTUDIO_TIMEOUT=30
# Agent scaffolding (optional):
ATLAS_GUARDRAILS=on                  # "off" = trust mode: don't auto-finalize or
                                     #   override the model's terminal choice
ATLAS_AGENT_DEBUG=                   # "1" = attach an intervention trace
                                     #   (debug.interventions) to chat responses
```

`ATLAS_GUARDRAILS` exists because the deterministic guardrails (auto-finalizing the
right reply after a plan change, correcting a wrong terminal choice) prop up a weak
local model. Turning them **off** plus reading the intervention trace is how you
measure a stronger model's *raw* reliability — and decide how much scaffolding it lets
you remove.

---

## What to try in the chat
- `What are the prerequisites for Machine Learning?` — regulatory lookup (shows the
  full AND/OR/k-of-n expression, not a flattened edge)
- `Can I take Operating Systems now?` — eligibility against your transcript
- `Why can't I take the capstone?` — the **actual** failed constraint, with its
  catalog citation (not a guessed reason)
- `What should I take next term?` — the planner's recommendation
- `What do I still need to graduate?` — requirement audit
- `Delay Operating Systems` / `Take Machine Learning earlier` — re-plans and shows a
  branch; requested changes are amber, forced downstream changes are rose
- `Balance my workload, I can graduate a year later` — re-plans for an **even** credit
  load across terms instead of the earliest possible graduation
- `I want to graduate a semester earlier` — infeasible at the normal cap, so the agent
  **offers the relaxation** (e.g. 21 or 24 credits/term) as acceptable branches
- Drag a course to another term on the board — spins off a new candidate branch
- `Compare the two plans` (or `compare c1 and c2`) — narrates the structured diff
- `Use this` / `Discard` on any branch — accept or drop it; several can be held at once

### How plan changes work
- **Compose, don't enumerate.** One `replan` tool takes any combination of constraints
  (pin + raise cap + graduate later + balance), so new requests don't need new tools.
- **Relax, don't dead-end.** Infeasible? The agent finds the minimal goal-preserving
  change that works and offers it. Only when nothing reasonable helps does it report a
  blocking reason (human-in-the-loop).
- **Hold and compare.** Alternatives are named branches (`c1`, `c2`, …) the student can
  keep side by side, compare, and accept individually — not a single overwrite slot.
- **Remember across turns.** Pending branches and the constraints behind them are
  carried forward, so "make it one semester later" refines the last balance instead of
  starting over.

---

## How the code maps to the architecture

| Component | File | Role |
|---|---|---|
| VPRS (logic-bearing requirement spec) | `backend/vprs.py` | The **UET–VNU B.Sc. Computer Science** program (135 credits, 62 courses): AND/OR/k-of-n prerequisites, per-block credit minimums, per-rule audit source. Compulsory courses are `REQUIRED`; elective blocks are expressed as category credit minimums (the planner selects which electives to take). Course offerings aren't in the official curriculum, so all are "both" (no season constraint). |
| Student state | `backend/student_data.py` | Immutable history + marks; `current_term`; a demo student with a failed-then-retaken course. |
| Planning agent | `backend/planner.py` | **OR-Tools CP-SAT** constraint scheduler (complete: finds a feasible plan if one exists, proves infeasibility otherwise) with a greedy fallback (`PLANNER=greedy`, or if OR-Tools is absent). Encodes AND/OR/k-of-n prerequisites, coreqs, offerings, and credit caps. Parameterized: per-term `credit_cap`, graduation bounds (`min`/`max_grad_term`), and a selectable objective — `early` / `even` (workload balance) / `minimal_change`. `relax()` searches minimal relaxations when a request is infeasible. Same `plan_path` interface for both backends. |
| Verifier | `backend/verifier.py` | Deterministic constraint checker; returns structured violations with citations; single-subject eligibility. Accepts a `credit_cap` override so a deliberate overload isn't flagged invalid. |
| Plan-diff | `backend/diff.py` | Structured comparison of two plans; splits **requested** vs **induced** changes; finds the fork term. |
| LLM client | `backend/llm_client.py` | Backend-agnostic Claude/LM Studio transport; tool-use helper (`chat_tools`, with an OpenAI-compatible adapter for local models) + parse/render with deterministic fallbacks. |
| Agent (orchestrator) | `backend/agent.py` | Tool-using loop. The model **observes** facts, calls the one composable **`replan`** tool to **infer** a plan change (or get relaxations), and **displays** the result. Holds multiple candidate branches, carries a cross-turn working-state briefing, and exposes switchable, observable guardrails (`ATLAS_GUARDRAILS`, `ATLAS_AGENT_DEBUG`). All tools are deterministic Python wrappers; the model only orchestrates. |
| Backend + fallback router | `backend/qa.py` | Deterministic resolvers + `simulate_change` + multi-candidate helpers (`add_candidate` / `candidate_view` / …). `handle` dispatches to the agent when a tool-capable backend is available, else to the legacy parse→route fallback (`handle_deterministic`). |
| API + server | `backend/app.py` | `/api/state` (returns all candidate branches), `/api/chat`, `/api/accept` & `/api/discard` (by branch `id`), `/api/reschedule` (drag-and-drop branch); serves the frontend. |
| Frontend | `frontend/index.html` | Branching timeline (history → trunk → N alternative branches), per-branch **Use / Compare / Discard**, drag-and-drop rescheduling, chat, comparison panel. No build step. |
| Tests | `backend/test_replan.py` | 48 deterministic tests (no model/network): constraint composition, relaxation search, cross-turn memory, multi-candidate state, and the agent guardrails via a scripted fake LLM. |

---

## Scope notes (honest)

This is the **pilot / product** slice (the "cố vấn học tập" advisor). It is deliberately
not the research contribution. In particular:

- **The planner is OR-Tools CP-SAT** — a complete constraint model that finds a
  feasible plan when one exists and proves infeasibility otherwise. A greedy backend
  remains behind the same `plan_path` interface as a fallback (`PLANNER=greedy`, or
  when OR-Tools isn't installed). The objective is **selectable per request**: `early`
  (graduate as soon as possible, then front-load), `even` (minimize the peak per-term
  credit load — soft workload balancing), or `minimal_change` (stay closest to the
  current plan). Graduation bounds and the per-term credit cap are parameters of the
  same model, so a request like "balance the load and I'll graduate a year later"
  composes into one solve.
- **The VPRS here is hand-encoded**, not ingested. The research contribution is the
  *ingestion* of messy course-outline prose into this verifiable, logic-bearing spec,
  evaluated on extraction fidelity against a hand-verified gold spec — that pipeline
  is not in this app.
- **Small edits are anchored to the baseline.** Under the default `minimal_change`
  objective, a what-if passes the current plan as an `anchor` to CP-SAT, which adds a
  minimal-deviation term: every course except the requested one stays in its baseline
  slot, so the branch shows only the requested change and its genuinely forced
  consequences (not a fresh elective pick). The `even` and `early` objectives
  intentionally re-pick across the plan, since balancing/compressing is the goal.
- **Infeasible requests are negotiated, not refused.** The agent searches a small,
  goal-preserving relaxation ladder (raise the cap, allow graduating a little later,
  drop a pin) and offers what works. A blocking reason is returned only when no
  reasonable relaxation helps — still human-in-the-loop, never a fabricated branch.
- **No efficacy claim.** The app validates that recommendations are *well-formed*
  against the spec and the student's record. It does not claim to improve outcomes.
- **No risk signal.** This stays at subject/program granularity, distinct from the
  risk-prediction work.
- **Session state is in-memory and single-user.** Candidate branches, plan, and
  cross-turn context live in one global session for a demo student; per-user storage
  is still TODO before multi-user deployment.
