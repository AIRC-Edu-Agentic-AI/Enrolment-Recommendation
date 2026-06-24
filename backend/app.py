"""
FastAPI server. Serves the frontend and exposes the advising API.
Session state is in-memory (single demo user); swap for per-user storage in prod.
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os

from vprs import PROGRAM, term_label
from student_data import demo_state
from planner import plan_path
import qa
import llm_client

app = FastAPI(title="ATLAS Course Advisor")

PROG = PROGRAM
STATE = demo_state()
SESSION = {"current_plan": None, "candidates": [], "last_constraints": None,
           "history": []}


def _ensure_plan():
    if not SESSION["current_plan"]:
        plan, reason = plan_path(PROG, STATE)
        SESSION["current_plan"] = plan
        SESSION["plan_reason"] = reason


class ChatIn(BaseModel):
    message: str


class RescheduleIn(BaseModel):
    code: str
    term: int


class CandidateRef(BaseModel):
    id: Optional[str] = None


def _candidate_views():
    plan = SESSION["current_plan"]
    return [qa.candidate_view(PROG, STATE, plan, rec)
            for rec in SESSION.get("candidates", [])]


@app.get("/api/state")
def get_state():
    _ensure_plan()
    plan = SESSION["current_plan"]
    return {
        "program": PROG.name,
        "student": STATE.name,
        "current_term": STATE.current_term,
        "history": STATE.to_dict(PROG),
        "current_plan": qa._plan_view(PROG, plan) if plan else None,
        "candidates": _candidate_views(),
        "llm_online": llm_client.llm_available(),
        "num_terms": PROG.num_terms,
        "term_labels": {t: term_label(t) for t in range(1, PROG.horizon + 1)},
    }


@app.post("/api/chat")
def chat(inp: ChatIn):
    _ensure_plan()
    resp = qa.handle(PROG, STATE, SESSION, inp.message)
    return resp


@app.post("/api/reschedule")
def reschedule(inp: RescheduleIn):
    """Drag-and-drop pin: move a course to a specific term, adding a candidate branch."""
    _ensure_plan()
    if inp.code not in PROG.subjects:
        return {"ok": False, "error": f"Unknown course {inp.code}", "candidates": None}
    action = {"type": "pin", "code": inp.code, "term": inp.term}
    sim = qa.simulate_change(PROG, STATE, SESSION["current_plan"], action)
    if not sim["feasible"]:
        return {"ok": False, "error": sim["reason"], "candidates": None}
    label = f"Move {PROG.subjects[inp.code].name} to {term_label(inp.term)}"
    qa.add_candidate(SESSION, sim["candidate"], requested=sim["requested"],
                     summary=sim["summary"], label=label)
    return {"ok": True, "text": sim["summary"], "candidates": _candidate_views()}


@app.post("/api/accept")
def accept(ref: CandidateRef = CandidateRef()):
    """Adopt a candidate branch as the current plan. Defaults to the latest if no id.
    Accepting one resolves the choice, so all pending branches are cleared."""
    rec = qa.get_candidate(SESSION, ref.id) if ref.id else qa.latest_candidate(SESSION)
    if not rec:
        return {"ok": False, "error": "no candidate to accept"}
    SESSION["current_plan"] = rec["plan"]
    qa.clear_candidates(SESSION)
    return {"ok": True, "current_plan": qa._plan_view(PROG, SESSION["current_plan"]),
            "candidates": _candidate_views()}


@app.post("/api/discard")
def discard(ref: CandidateRef = CandidateRef()):
    """Discard one candidate branch by id, or all of them if no id is given."""
    if ref.id:
        SESSION["candidates"] = [r for r in SESSION.get("candidates", [])
                                 if r["id"] != ref.id]
        if not SESSION["candidates"]:
            SESSION["last_constraints"] = None
    else:
        qa.clear_candidates(SESSION)
    return {"ok": True, "candidates": _candidate_views()}


# serve frontend
FRONT = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONT, "index.html"))


if os.path.isdir(FRONT):
    app.mount("/static", StaticFiles(directory=FRONT), name="static")
