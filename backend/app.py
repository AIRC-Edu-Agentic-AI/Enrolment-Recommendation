"""
FastAPI server. Serves the frontend and exposes the advising API.
Session state is in-memory (single demo user); swap for per-user storage in prod.
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from vprs import PROGRAM, term_label
from student_data import demo_state
from planner import plan_path
import qa
import llm_client

app = FastAPI(title="ATLAS Course Advisor")

PROG = PROGRAM
STATE = demo_state()
SESSION = {"current_plan": None, "candidate": None, "candidate_requested": [],
           "history": []}


def _ensure_plan():
    if not SESSION["current_plan"]:
        plan, reason = plan_path(PROG, STATE)
        SESSION["current_plan"] = plan
        SESSION["plan_reason"] = reason


class ChatIn(BaseModel):
    message: str


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
        "candidate": qa._plan_view(PROG, SESSION["candidate"]) if SESSION["candidate"] else None,
        "llm_online": llm_client.llm_available(),
        "num_terms": PROG.num_terms,
        "term_labels": {t: term_label(t) for t in range(1, PROG.horizon + 1)},
    }


@app.post("/api/chat")
def chat(inp: ChatIn):
    _ensure_plan()
    resp = qa.handle(PROG, STATE, SESSION, inp.message)
    return resp


@app.post("/api/accept")
def accept():
    if SESSION["candidate"]:
        SESSION["current_plan"] = SESSION["candidate"]
        SESSION["candidate"] = None
        SESSION["candidate_requested"] = []
        return {"ok": True, "current_plan": qa._plan_view(PROG, SESSION["current_plan"])}
    return {"ok": False, "error": "no candidate to accept"}


@app.post("/api/discard")
def discard():
    SESSION["candidate"] = None
    SESSION["candidate_requested"] = []
    return {"ok": True}


# serve frontend
FRONT = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONT, "index.html"))


if os.path.isdir(FRONT):
    app.mount("/static", StaticFiles(directory=FRONT), name="static")
