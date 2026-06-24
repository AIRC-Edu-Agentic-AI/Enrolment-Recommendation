"""
Student state model. History is immutable (completed terms + marks). `current_term`
is the next term to plan. The live 80/20 ingestion in the research design maps onto
"new marks arrive -> state updates -> planner re-plans"; here we expose a simple
in-memory state with a method to post a new term's results.
"""

PASS_THRESHOLD = 5.0  # out of 10


class StudentState:
    def __init__(self, name, history):
        # history: {term_index: [(code, mark), ...]}
        self.name = name
        self.history = {int(t): list(v) for t, v in history.items()}

    @property
    def current_term(self):
        return (max(self.history.keys()) + 1) if self.history else 1

    def passed_marks(self):
        """code -> mark, for subjects whose mark >= threshold."""
        out = {}
        for term, recs in self.history.items():
            for code, mark in recs:
                if mark >= PASS_THRESHOLD:
                    out[code] = mark
        return out

    def failed(self):
        out = []
        for term, recs in self.history.items():
            for code, mark in recs:
                if mark < PASS_THRESHOLD:
                    out.append((code, mark, term))
        return out

    def post_term(self, term, results):
        self.history[int(term)] = list(results)

    def to_dict(self, prog):
        terms = []
        for t in sorted(self.history.keys()):
            terms.append({
                "term": t,
                "subjects": [
                    {"code": c, "name": prog.subjects[c].name if c in prog.subjects else c,
                     "credits": prog.subjects[c].credits if c in prog.subjects else 0,
                     "category": prog.subjects[c].category if c in prog.subjects else "",
                     "mark": m, "passed": m >= PASS_THRESHOLD}
                    for c, m in self.history[t]],
            })
        return terms


# A demo student in the UET-VNU B.Sc. CS program: 3 terms done, with one failed-
# then-retaken subject (UET.CS2046) to make re-planning interesting. current_term=4.
DEMO_HISTORY = {
    1: [("PHI1006", 7.5), ("UET.MAT1050", 6.0), ("UET.MAT1053", 7.0),
        ("UET.COM1050", 8.0), ("FLF1107", 6.5)],
    2: [("PEC1008", 6.0), ("UET.MAT1051", 5.5), ("UET.MAT1052", 6.0),
        ("UET.CS1058", 7.0), ("UET.PHY1095", 5.5)],
    3: [("UET.CS2043", 7.0), ("UET.MAT1057", 6.0), ("UET.CS2046", 4.0),
        ("VNU1001", 6.5), ("UET.PHY1096", 6.0)],
    #                       ^ UET.CS2046 (Artificial Intelligence) failed (4.0)
}


def demo_state():
    return StudentState("Demo Student", DEMO_HISTORY)
