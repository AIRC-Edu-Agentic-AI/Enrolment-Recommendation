# ATLAS Course Advisor (pilot)

A course-registration advising app for a bachelor program. The student sees their
enrollment history and current inferred plan; they chat to ask questions or request
a change, and the app shows the new plan as a **branch** off the current one, with
**requested** vs **forced (induced)** changes colored differently and a structured
comparison.

The LLM is a **tool-using agent**, not a knowledge source. For each question it decides
which tools to call to **observe** verified facts (prerequisites, eligibility, audit),
**infer** plan changes (what-if simulation, comparison), and **display** the result
(a plain answer, a plan branch, an infeasible note, or a clarifying question). Every
regulatory and planning fact — prerequisites, credit math, what's feasible, why
something is blocked — is computed by deterministic Python against a logic-bearing
requirement spec; the tools wrap that backend, so the model orchestrates but never
invents facts. If a tool-capable model is offline (or `LLM_PROVIDER=lmstudio`), the app
falls back to a rule-based parser + template renderer and stays fully usable.

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

### Configuration
All settings live in `.env` (read by `backend/llm_client.py` via python-dotenv):
```bash
LLM_PROVIDER=anthropic               # "anthropic" (default) or "lmstudio"
ANTHROPIC_API_KEY=sk-ant-...         # required for the anthropic provider
ANTHROPIC_MODEL=claude-haiku-4-5     # model to call
# Local-model alternative:
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
LMSTUDIO_TIMEOUT=30
```

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
- `Compare the two plans` — narrates the structured diff
- Infeasible asks (e.g. pinning a course before its prerequisite) return a blocking
  reason and **no fake branch** (human-in-the-loop)

---

## How the code maps to the architecture

| Component | File | Role |
|---|---|---|
| VPRS (logic-bearing requirement spec) | `backend/vprs.py` | The **UET–VNU B.Sc. Computer Science** program (135 credits, 62 courses): AND/OR/k-of-n prerequisites, per-block credit minimums, per-rule audit source. Compulsory courses are `REQUIRED`; elective blocks are expressed as category credit minimums (the planner selects which electives to take). Course offerings aren't in the official curriculum, so all are "both" (no season constraint). |
| Student state | `backend/student_data.py` | Immutable history + marks; `current_term`; a demo student with a failed-then-retaken course. |
| Planning agent | `backend/planner.py` | **OR-Tools CP-SAT** constraint scheduler (complete: finds a feasible plan if one exists, proves infeasibility otherwise) with a greedy fallback (`PLANNER=greedy`, or if OR-Tools is absent). Encodes AND/OR/k-of-n prerequisites, coreqs, offerings, and credit caps; accepts what-if modifiers (delay / earlier / drop / pin). Same `plan_path` interface for both backends. |
| Verifier | `backend/verifier.py` | Deterministic constraint checker; returns structured violations with citations; single-subject eligibility. |
| Plan-diff | `backend/diff.py` | Structured comparison of two plans; splits **requested** vs **induced** changes; finds the fork term. |
| LLM client | `backend/llm_client.py` | Backend-agnostic Claude/LM Studio transport; tool-use helper (`chat_tools`) + parse/render with deterministic fallbacks. |
| Agent (orchestrator) | `backend/agent.py` | Tool-using loop. Claude decides which tools to **observe** (facts), **infer** (plans), and **display** (reply / branch / clarify). All tools are deterministic Python wrappers; the model only orchestrates. |
| Backend + fallback router | `backend/qa.py` | Deterministic resolvers + `simulate_change` (shared by the agent's tools). `handle` dispatches to the agent when a tool-capable backend is available, else to the legacy parse→route fallback (`handle_deterministic`). |
| API + server | `backend/app.py` | `/api/state`, `/api/chat`, `/api/accept`, `/api/discard`; serves the frontend. |
| Frontend | `frontend/index.html` | Branching timeline (history → trunk → alternative), chat, comparison panel. No build step. |

---

## Scope notes (honest)

This is the **pilot / product** slice (the "cố vấn học tập" advisor). It is deliberately
not the research contribution. In particular:

- **The planner is OR-Tools CP-SAT** — a complete constraint model that finds a
  feasible plan when one exists and proves infeasibility otherwise. A greedy backend
  remains behind the same `plan_path` interface as a fallback (`PLANNER=greedy`, or
  when OR-Tools isn't installed). Objective: graduate as early as possible, then honor
  "take X earlier", then front-load. It is not yet multi-objective tuned (no soft
  load-balancing / min-credit preferences).
- **The VPRS here is hand-encoded**, not ingested. The research contribution is the
  *ingestion* of messy course-outline prose into this verifiable, logic-bearing spec,
  evaluated on extraction fidelity against a hand-verified gold spec — that pipeline
  is not in this app.
- **What-if re-solves are anchored to the baseline.** Each what-if passes the current
  plan as an `anchor` to CP-SAT, which adds a minimal-deviation objective: every course
  except the requested one is kept in its baseline slot, so the branch shows only the
  requested change and its genuinely forced consequences (not a fresh elective pick).
  The requested course itself is governed by its modifier (pin term / delay / earliest).
- **No efficacy claim.** The app validates that recommendations are *well-formed*
  against the spec and the student's record. It does not claim to improve outcomes.
- **No risk signal.** This stays at subject/program granularity, distinct from the
  risk-prediction work.
- Session state is in-memory for a single demo student; swap for per-user storage to
  deploy.
