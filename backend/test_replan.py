"""
Deterministic test suite for the unified constraint-based re-planner.

The redesign collapsed three bespoke tools (simulate_plan_change,
plan_with_credit_override, rebalance_workload) into ONE `replan` capability over an
open constraint set, and parameterized the CP-SAT objective (credit cap, graduation
bounds, objective mode) instead of hardcoding three Minimize() branches. These tests
prove the new contract end-to-end at the deterministic core — no LLM involved, so
they are fast and reproducible. The LLM's job (mapping prose -> this constraint set)
is verified separately against the live model.

Two layers are covered:
  * planner.plan_path  — does the solver honor each constraint, and do they COMPOSE?
  * agent._Ctx.replan  — does the agent assemble prose-shaped args into constraints,
                         translate term labels, and never mis-flag a deliberate
                         overload as invalid?

Run:  python -m unittest test_replan -v
"""
import unittest

from vprs import PROGRAM, term_label
from student_data import demo_state
from planner import plan_path, plan_credits, grad_term, relax
from verifier import verify_plan
import agent
import qa


def _course_term(plan, code):
    for t, codes in plan.items():
        if code in codes:
            return t
    return None


def _peak_future(prog, plan, current_term):
    loads = [v for t, v in plan_credits(prog, plan).items() if t >= current_term]
    return max(loads) if loads else 0


class PlannerConstraintTests(unittest.TestCase):
    """The CP-SAT core: every constraint is honored, and constraints compose."""

    @classmethod
    def setUpClass(cls):
        cls.prog = PROGRAM
        cls.state = demo_state()
        cls.cur = cls.state.current_term
        cls.base, reason = plan_path(cls.prog, cls.state)
        assert reason is None, f"baseline infeasible: {reason}"
        cls.base_grad = grad_term(cls.base)
        cls.base_peak = _peak_future(cls.prog, cls.base, cls.cur)

    def test_baseline_is_valid_and_on_time(self):
        # A fresh plan should be violation-free and graduate by the nominal length.
        self.assertEqual(verify_plan(self.prog, self.state, self.base), [])
        self.assertLessEqual(self.base_grad, self.prog.num_terms)
        self.assertLessEqual(self.base_peak, self.prog.max_credits_per_term)

    def test_credit_cap_override_enables_earlier_graduation(self):
        # Raising the per-term cap should let the planner graduate no later — and
        # in this program, strictly earlier — by overloading some terms.
        cand, reason = plan_path(self.prog, self.state,
                                 modifiers={"credit_cap": 24}, anchor=self.base)
        self.assertIsNone(reason)
        self.assertLess(grad_term(cand), self.base_grad)
        self.assertGreater(_peak_future(self.prog, cand, self.cur),
                           self.prog.max_credits_per_term)

    def test_min_grad_term_forces_later_graduation(self):
        # "Graduate later": pinning a lower bound makes graduation land exactly there.
        target = self.base_grad + 1
        cand, reason = plan_path(self.prog, self.state,
                                 modifiers={"min_grad_term": target}, anchor=self.base)
        self.assertIsNone(reason)
        self.assertEqual(grad_term(cand), target)

    def test_max_grad_term_feasible_is_respected(self):
        cand, reason = plan_path(self.prog, self.state,
                                 modifiers={"max_grad_term": self.base_grad},
                                 anchor=self.base)
        self.assertIsNone(reason)
        self.assertLessEqual(grad_term(cand), self.base_grad)

    def test_max_grad_term_too_early_is_infeasible(self):
        # Demanding graduation well before the requirements can fit -> a reason, no plan.
        cand, reason = plan_path(self.prog, self.state,
                                 modifiers={"max_grad_term": self.cur},
                                 anchor=self.base)
        self.assertIsNotNone(reason)
        self.assertEqual(cand, {})

    def test_even_objective_reduces_peak_load(self):
        # With graduation allowed to slip, the 'even' objective must spread credits
        # so the peak term load drops below the early-graduation plan's peak.
        cand, reason = plan_path(
            self.prog, self.state,
            modifiers={"objective": "even", "max_grad_term": self.base_grad + 2},
            anchor=self.base)
        self.assertIsNone(reason)
        self.assertLess(_peak_future(self.prog, cand, self.cur), self.base_peak)

    def test_pin_places_course_in_requested_term(self):
        late_course = sorted(self.base[self.base_grad])[0]
        target = self.base_grad + 1
        cand, reason = plan_path(self.prog, self.state,
                                 modifiers={"pin": {late_course: target}},
                                 anchor=self.base)
        self.assertIsNone(reason)
        self.assertEqual(_course_term(cand, late_course), target)

    def test_drop_excludes_course(self):
        # Drop an elective (the planner can satisfy the category minimum with a
        # different one, so the plan stays feasible) and assert it is gone.
        elective = next((c for codes in self.base.values() for c in codes
                         if c not in self.prog.required_codes), None)
        self.assertIsNotNone(elective, "baseline has no elective to drop")
        cand, reason = plan_path(self.prog, self.state,
                                 modifiers={"drop": {elective}}, anchor=self.base)
        self.assertIsNone(reason)
        self.assertIsNone(_course_term(cand, elective))

    def test_composition_pin_and_credit_cap(self):
        # The whole point of the redesign: two constraints in ONE solve, both honored.
        late_course = sorted(self.base[self.base_grad])[0]
        target = self.base_grad + 1
        cand, reason = plan_path(
            self.prog, self.state,
            modifiers={"pin": {late_course: target}, "credit_cap": 21},
            anchor=self.base)
        self.assertIsNone(reason)
        self.assertEqual(_course_term(cand, late_course), target)
        # cap honored
        for t, load in plan_credits(self.prog, cand).items():
            self.assertLessEqual(load, 21)

    def test_composition_even_and_graduate_later(self):
        # "Balance the workload and I'll graduate a year later" -> both effects.
        target = self.base_grad + 2
        cand, reason = plan_path(
            self.prog, self.state,
            modifiers={"objective": "even", "min_grad_term": target,
                       "max_grad_term": target},
            anchor=self.base)
        self.assertIsNone(reason)
        self.assertEqual(grad_term(cand), target)
        self.assertLess(_peak_future(self.prog, cand, self.cur), self.base_peak)


class CtxReplanTests(unittest.TestCase):
    """The agent layer: prose-shaped args -> constraints, term-label translation,
    and correct feasibility/validity reporting."""

    def setUp(self):
        self.session = {}
        self.ctx = agent._Ctx(PROGRAM, demo_state(), self.session)
        self.cur = self.ctx.state.current_term
        self.base_grad = max(self.ctx.base.keys())

    def test_parse_term_label(self):
        self.assertEqual(agent._Ctx._parse_term_label("Y5 Fall"), 9)
        self.assertEqual(agent._Ctx._parse_term_label("Y3 Spring"), 6)
        self.assertEqual(agent._Ctx._parse_term_label("Y1 Fall"), 1)
        self.assertEqual(agent._Ctx._parse_term_label("Y5"), 9)   # defaults to Fall
        self.assertIsNone(agent._Ctx._parse_term_label("nonsense"))
        self.assertIsNone(agent._Ctx._parse_term_label(None))

    def test_pin_via_term_label_lands_in_year_not_term(self):
        # Regression: "Y5 Fall" must mean year 5 (term 9), never term 5.
        late_course = sorted(self.ctx.base[self.base_grad])[0]
        out = self.ctx.replan(pin=[{"course": late_course, "term_label": "Y5 Fall"}])
        self.assertTrue(out["feasible"])
        self.assertEqual(_course_term(self.ctx.last_sim["candidate"], late_course), 9)

    def test_graduate_later_moves_graduation_later(self):
        # Regression: "graduate one term later" must move graduation LATER, not earlier.
        target_label = term_label(self.base_grad + 1)
        out = self.ctx.replan(graduate_no_earlier_than=target_label)
        self.assertTrue(out["feasible"])
        self.assertEqual(out["graduation_term"], target_label)
        self.assertEqual(grad_term(self.ctx.last_sim["candidate"]), self.base_grad + 1)

    def test_graduate_later_by_relative_delta(self):
        # The model-friendly relative form: "one year later" -> graduate_later_by=2,
        # resolved against the current graduation term with no lookup needed.
        out = self.ctx.replan(graduate_later_by=2)
        self.assertTrue(out["feasible"])
        self.assertEqual(grad_term(self.ctx.last_sim["candidate"]), self.base_grad + 2)

    def test_graduate_later_alone_rebalances_not_one_course_shove(self):
        # The fix: volunteering to graduate later should buy a lighter load, not a
        # minimal one-course slip that leaves the peak unchanged.
        base_peak = _peak_future(PROGRAM, self.ctx.base, self.cur)
        out = self.ctx.replan(graduate_later_by=1)
        self.assertTrue(out["feasible"])
        self.assertLess(out["peak_credits_per_term"], base_peak)
        # graduation still lands one term later, as asked
        self.assertEqual(grad_term(self.ctx.last_sim["candidate"]), self.base_grad + 1)

    def test_explicit_minimal_change_still_available(self):
        # The student (or model) can still force a minimal delay by naming the objective.
        out = self.ctx.replan(graduate_later_by=1, objective="minimal_change")
        self.assertTrue(out["feasible"])
        # minimal_change keeps the peak at the baseline (no rebalance)
        base_peak = _peak_future(PROGRAM, self.ctx.base, self.cur)
        self.assertEqual(out["peak_credits_per_term"], base_peak)

    def test_replan_records_constraints_for_continuity(self):
        out = self.ctx.replan(objective="even", graduate_later_by=2)
        self.assertTrue(out["feasible"])
        desc = self.ctx.last_sim.get("constraints")
        self.assertIsNotNone(desc)
        self.assertIn("objective=even", desc)

    def test_credit_override_not_flagged_invalid(self):
        # Regression: a deliberate overload must NOT be reported as an invalid plan.
        out = self.ctx.replan(credit_cap=24)
        self.assertTrue(out["feasible"])
        self.assertTrue(self.ctx.last_sim["diff"]["valid_b"])
        self.assertLess(grad_term(self.ctx.last_sim["candidate"]), self.base_grad)

    def test_even_objective_lowers_peak(self):
        base_peak = _peak_future(PROGRAM, self.ctx.base, self.cur)
        out = self.ctx.replan(objective="even",
                              graduate_no_earlier_than=term_label(self.base_grad + 2))
        self.assertTrue(out["feasible"])
        self.assertLess(out["peak_credits_per_term"], base_peak)

    def test_combined_pin_and_even(self):
        # Compose a pin with a workload objective in a single replan call.
        late_course = sorted(self.ctx.base[self.base_grad])[0]
        out = self.ctx.replan(
            pin=[{"course": late_course, "term_label": term_label(self.base_grad + 2)}],
            objective="even",
            graduate_no_earlier_than=term_label(self.base_grad + 2))
        self.assertTrue(out["feasible"])
        cand = self.ctx.last_sim["candidate"]
        self.assertEqual(_course_term(cand, late_course), self.base_grad + 2)

    def test_system_prompt_renders(self):
        # The prompt is an f-string; literal {course:...} examples must be escaped or
        # agent.run raises on every request and silently falls back to the
        # deterministic router. Guard against that regression.
        prompt = agent._system_prompt(PROGRAM, demo_state(), {})
        self.assertIn("replan", prompt)
        self.assertIn(PROGRAM.name, prompt)

    def test_unknown_course_is_reported_not_crashed(self):
        out = self.ctx.replan(drop=["NOPE.404"])
        self.assertFalse(out["feasible"])
        self.assertIn("unknown course", out["reason"])

    def test_requested_courses_tagged_requested_in_diff(self):
        # A pinned course should appear as a REQUESTED change, not an induced one.
        late_course = sorted(self.ctx.base[self.base_grad])[0]
        self.ctx.replan(pin=[{"course": late_course,
                              "term_label": term_label(self.base_grad + 1)}])
        req_codes = {c["code"] for c in self.ctx.last_sim["diff"]["requested_changes"]}
        self.assertIn(late_course, req_codes)


class SessionContinuityTests(unittest.TestCase):
    """P1: cross-turn working-state memory. The pending candidate and the constraints
    behind it must survive into the next turn's briefing, and the briefing must carry
    the prior intent so a follow-up can refine it (no LLM involved here — we drive the
    deterministic pieces directly)."""

    def _finalize_present(self, ctx):
        # Mimic the loop's auto-finalize: commit the pending candidate to session.
        return ctx.finalize("present_plan_change",
                            {"text": ctx.last_sim.get("summary", "")})

    def test_briefing_absent_with_no_pending_state(self):
        self.assertIsNone(agent._session_briefing({}))

    def test_replan_then_present_populates_session_state(self):
        session = {}
        ctx = agent._Ctx(PROGRAM, demo_state(), session)
        ctx.replan(objective="even", graduate_later_by=2)
        self._finalize_present(ctx)
        self.assertEqual(len(session.get("candidates", [])), 1)
        self.assertIn("objective=even", session.get("last_constraints", ""))

    def test_briefing_surfaces_pending_plan_and_constraints(self):
        session = {}
        ctx = agent._Ctx(PROGRAM, demo_state(), session)
        ctx.replan(objective="even", graduate_later_by=2)
        self._finalize_present(ctx)
        brief = agent._session_briefing(session)
        self.assertIsNotNone(brief)
        self.assertIn("[Session context]", brief)
        self.assertIn("pending", brief.lower())
        self.assertIn("objective=even", brief)        # prior intent carried forward

    def test_infeasible_does_not_destroy_held_candidates(self):
        # P3 semantics: a later infeasible request must NOT wipe branches the student
        # is already holding — they remain valid options.
        session = {}
        ctx = agent._Ctx(PROGRAM, demo_state(), session)
        ctx.replan(objective="even", graduate_later_by=2)
        self._finalize_present(ctx)
        self.assertEqual(len(session["candidates"]), 1)
        ctx2 = agent._Ctx(PROGRAM, demo_state(), session)
        ctx2.last_sim = {"feasible": False, "reason": "nope"}
        env = ctx2.finalize("report_infeasible", {"text": "nope"})
        self.assertTrue(env["infeasible"])
        self.assertEqual(len(session["candidates"]), 1)   # untouched


class RelaxationTests(unittest.TestCase):
    """P2: an infeasible request is not a dead-end. The relaxation search must return
    minimal, GOAL-PRESERVING changes that are genuinely feasible, and the agent must
    surface them as an acceptable branch."""

    @classmethod
    def setUpClass(cls):
        cls.prog = PROGRAM
        cls.state = demo_state()
        cls.base, _ = plan_path(cls.prog, cls.state)
        cls.base_grad = grad_term(cls.base)

    def test_earlier_graduation_is_infeasible_at_default_cap(self):
        # Premise for the rest: graduating a term early genuinely doesn't fit at 18.
        mods = {"max_grad_term": self.base_grad - 1}
        _, reason = plan_path(self.prog, self.state, modifiers=mods, anchor=self.base)
        self.assertIsNotNone(reason)

    def test_relax_returns_feasible_goal_preserving_options(self):
        mods = {"max_grad_term": self.base_grad - 1}
        alts = relax(self.prog, self.state, self.base, mods)
        self.assertGreaterEqual(len(alts), 1)
        for a in alts:
            # Each option is a REAL, valid plan that still meets the earlier deadline...
            cap = a["modifiers"].get("credit_cap")
            self.assertEqual(verify_plan(self.prog, self.state, a["plan"], credit_cap=cap), [])
            self.assertLessEqual(grad_term(a["plan"]), self.base_grad - 1)
            # ...and never the self-defeating "graduate later" for an EARLIER request.
            self.assertNotIn("later", a["description"])

    def test_relax_empty_when_truly_impossible(self):
        # Graduating absurdly early can't be salvaged by any single relaxation.
        mods = {"max_grad_term": self.state.current_term}
        self.assertEqual(relax(self.prog, self.state, self.base, mods), [])

    def test_replan_earlier_surfaces_alternatives(self):
        ctx = agent._Ctx(self.prog, demo_state(), {})
        out = ctx.replan(graduate_earlier_by=1)
        self.assertFalse(out["feasible"])
        self.assertTrue(out.get("alternatives"))
        # the cheapest option overloads (peak above the normal cap)
        self.assertGreater(out["alternatives"][0]["peak_credits_per_term"],
                           self.prog.max_credits_per_term)

    def test_offer_alternatives_adds_each_as_candidate(self):
        session = {}
        ctx = agent._Ctx(self.prog, demo_state(), session)
        out = ctx.replan(graduate_earlier_by=1)
        n_alts = len(out["alternatives"])
        env = ctx.finalize("offer_alternatives", {})
        self.assertEqual(env["structured"]["kind"], "relaxation")
        self.assertIsNotNone(env["candidate"])
        # every relaxation becomes a holdable branch (P3), capped at MAX_CANDIDATES
        self.assertEqual(len(session["candidates"]), min(n_alts, qa.MAX_CANDIDATES))
        # each offered branch really does graduate earlier than the baseline
        for rec in session["candidates"]:
            self.assertLessEqual(grad_term(rec["plan"]), self.base_grad - 1)

    def test_impossible_request_falls_back_to_infeasible(self):
        session = {}
        ctx = agent._Ctx(self.prog, demo_state(), session)
        out = ctx.replan(graduate_earlier_by=10)
        self.assertFalse(out["feasible"])
        self.assertFalse(out.get("alternatives"))
        env = ctx.finalize("offer_alternatives", {})  # degrades gracefully
        self.assertTrue(env.get("infeasible"))


class MultiCandidateTests(unittest.TestCase):
    """P3: the student can hold, compare, and accept several alternative branches
    instead of a single overwrite-on-each-change slot."""

    def setUp(self):
        self.state = demo_state()
        self.session = {}
        self.ctx = agent._Ctx(PROGRAM, self.state, self.session)
        self.base_grad = max(self.ctx.base.keys())

    def _present(self, **kwargs):
        self.ctx.replan(**kwargs)
        return self.ctx.finalize("present_plan_change",
                                {"text": self.ctx.last_sim.get("summary", "")})

    def test_branches_accumulate_not_overwrite(self):
        self._present(graduate_later_by=1)
        self._present(objective="even", graduate_later_by=2)
        self.assertEqual(len(self.session["candidates"]), 2)
        ids = [c["id"] for c in self.session["candidates"]]
        self.assertEqual(len(set(ids)), 2)            # distinct ids

    def test_fifo_cap_at_max(self):
        for n in (1, 2, 1, 2, 1):                      # five changes
            self._present(graduate_later_by=(n))
        self.assertEqual(len(self.session["candidates"]), qa.MAX_CANDIDATES)

    def test_get_and_latest_candidate(self):
        self._present(graduate_later_by=1)
        self._present(graduate_later_by=2)
        first_id = self.session["candidates"][0]["id"]
        self.assertEqual(qa.get_candidate(self.session, first_id)["id"], first_id)
        self.assertEqual(qa.latest_candidate(self.session)["id"],
                         self.session["candidates"][-1]["id"])
        self.assertIsNone(qa.get_candidate(self.session, "nope"))

    def test_candidate_view_shape(self):
        self._present(graduate_later_by=1)
        view = qa.candidate_view(PROGRAM, self.state, self.ctx.base,
                                 self.session["candidates"][0])
        for key in ("id", "label", "plan", "diff", "grad_term", "peak"):
            self.assertIn(key, view)
        self.assertIn("terms", view["plan"])

    def test_compare_two_candidates_by_id(self):
        self._present(graduate_later_by=1)
        self._present(graduate_later_by=2)
        a, b = (c["id"] for c in self.session["candidates"][:2])
        out = self.ctx.compare_plans(plan_a=a, plan_b=b)
        self.assertIn("summary", out)
        self.assertNotIn("error", out)

    def test_compare_defaults_to_current_vs_latest(self):
        self._present(graduate_later_by=1)
        out = self.ctx.compare_plans()
        self.assertIn("summary", out)

    def test_briefing_lists_all_candidate_ids(self):
        self._present(graduate_later_by=1)
        self._present(objective="even", graduate_later_by=2)
        brief = agent._session_briefing(self.session)
        for c in self.session["candidates"]:
            self.assertIn(c["id"], brief)


if __name__ == "__main__":
    unittest.main(verbosity=2)
