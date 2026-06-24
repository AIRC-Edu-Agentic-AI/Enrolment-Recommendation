"""
Verifiable Program Requirement Specification (VPRS).

The logic-bearing core of the system. Prerequisites are NOT plain edges; they are
expression trees supporting AND / OR / k-of-n, so the spec stays faithful to how
real catalogs state requirements ("A and one of {B, C}", "two of {...}").

Every requirement carries a `source` string: the catalog/outline sentence it was
extracted from. This is the audit trail that makes the spec re-checkable and makes
"why not" answers cite a real regulation instead of a model guess.
"""
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Logical prerequisite expressions
# ---------------------------------------------------------------------------
# An expression is one of:
#   {"subject": "CS101"}                         -> atom
#   {"op": "AND", "args": [expr, ...]}
#   {"op": "OR",  "args": [expr, ...]}
#   {"op": "KOF", "k": 2, "args": [expr, ...]}   -> at least k of n
#   None                                          -> no prerequisite


def eval_prereq(expr, completed: set) -> bool:
    """True if `completed` (a set of subject codes) satisfies the expression."""
    if not expr:
        return True
    if "subject" in expr:
        return expr["subject"] in completed
    op = expr["op"]
    args = expr["args"]
    if op == "AND":
        return all(eval_prereq(a, completed) for a in args)
    if op == "OR":
        return any(eval_prereq(a, completed) for a in args)
    if op == "KOF":
        return sum(1 for a in args if eval_prereq(a, completed)) >= expr["k"]
    raise ValueError(f"unknown op {op}")


def prereq_atoms(expr) -> set:
    """All subject codes mentioned anywhere in the expression (for graph/blocking)."""
    if not expr:
        return set()
    if "subject" in expr:
        return {expr["subject"]}
    out = set()
    for a in expr["args"]:
        out |= prereq_atoms(a)
    return out


def describe_prereq(expr, names: dict) -> str:
    """Human-readable rendering of a prerequisite expression."""
    if not expr:
        return "none"
    if "subject" in expr:
        c = expr["subject"]
        return f"{c} ({names.get(c, c)})"
    op = expr["op"]

    def wrap(a):
        s = describe_prereq(a, names)
        # parenthesize any compound child to keep precedence unambiguous
        if a and "op" in a:
            return f"({s})"
        return s

    if op == "AND":
        return " and ".join(wrap(a) for a in expr["args"])
    if op == "OR":
        return " or ".join(wrap(a) for a in expr["args"])
    if op == "KOF":
        return (f"at least {expr['k']} of: "
                + ", ".join(describe_prereq(a, names) for a in expr["args"]))
    return "?"


# ---------------------------------------------------------------------------
# Subject + Program
# ---------------------------------------------------------------------------
@dataclass
class Subject:
    code: str
    name: str
    credits: int
    category: str
    offered: str               # "fall" | "spring" | "both"
    prereq: Optional[dict] = None
    coreq: list = field(default_factory=list)
    source: str = ""           # audit trail: the catalog sentence


@dataclass
class Program:
    name: str
    subjects: dict             # code -> Subject
    category_min_credits: dict # category -> min credits required to graduate
    required_codes: list       # the canonical required subject set (the degree)
    total_credits_required: int
    num_terms: int = 8         # nominal program length (on-time graduation target)
    max_credits_per_term: int = 18
    min_credits_per_term: int = 12
    max_terms: int = 0         # planning horizon (max time-to-degree); 0 -> num_terms

    def names(self):
        return {c: s.name for c, s in self.subjects.items()}

    @property
    def horizon(self):
        """Latest term the planner may schedule into. The objective still minimizes
        graduation, so plans finish by num_terms when feasible; the extra terms exist
        so a delayed what-if can show a *later* graduation instead of going infeasible."""
        return self.max_terms or self.num_terms


def season_of(term_index: int) -> str:
    """Odd term = fall, even term = spring (Year1 Fall = term 1)."""
    return "fall" if term_index % 2 == 1 else "spring"


def offered_in(subject: Subject, term_index: int) -> bool:
    if subject.offered == "both":
        return True
    return subject.offered == season_of(term_index)


def term_label(term_index: int) -> str:
    year = (term_index + 1) // 2
    return f"Y{year} {'Fall' if season_of(term_index)=='fall' else 'Spring'}"


# ---------------------------------------------------------------------------
# Program: B.Sc. Computer Science — UET, VNU Hanoi (135 credits)
# ---------------------------------------------------------------------------
# Offerings: the official curriculum does not state which semester each course
# runs, so every course is marked "both" (no season constraint). The planner is
# driven purely by prerequisites and credit caps.
#
# Elective blocks ("choose N credits of group G") are modelled as category credit
# minimums (see CATEGORY_MIN): compulsory courses sit in REQUIRED, electives are
# optional subjects in the same category, and the CP-SAT planner selects enough of
# them to meet each minimum. The non-credit graduation conditions (Physical
# Education, National Defence, Soft Skills) are excluded from the credit model.
def _S(code, name, credits, category, prereq=None, coreq=None, source=""):
    return Subject(code, name, credits, category, "both", prereq, coreq or [], source)


def _and(*codes):
    return {"op": "AND", "args": [{"subject": c} for c in codes]}


def _or(*codes):
    return {"op": "OR", "args": [{"subject": c} for c in codes]}


def _req(*codes):
    return "Prerequisite: " + ", ".join(codes) + "."


# -- I.1 General education — VNU compulsory block (21 credits) ----------------
_GEN_VNU = [
    _S("PHI1006", "Marxist-Leninist Philosophy", 3, "General (VNU)"),
    _S("PEC1008", "Marxist-Leninist Political Economy", 2, "General (VNU)",
       {"subject": "PHI1006"}, [], _req("PHI1006")),
    _S("HIS1001", "History of the Vietnam Communist Party", 2, "General (VNU)"),
    _S("POL1001", "Ho Chi Minh Ideology", 2, "General (VNU)"),
    _S("PHI1002", "Scientific Socialism", 2, "General (VNU)"),
    _S("FLF1107", "English B1", 5, "General (VNU)"),
    _S("VNU1001", "Introduction to Digital Technology and AI Applications", 3,
       "General (VNU)"),
    _S("THL1057", "State and Law", 2, "General (VNU)"),
]

# -- I.2 General education — by field (35 credits) ---------------------------
_GEN_FIELD = [
    _S("UET.MAT1053", "Linear Algebra for Engineers", 5, "General (Field)"),
    _S("UET.MAT1050", "Calculus 1", 5, "General (Field)"),
    _S("UET.MAT1051", "Calculus 2", 5, "General (Field)",
       {"subject": "UET.MAT1050"}, [], _req("UET.MAT1050")),
    _S("UET.PHY1095", "General Physics 1", 3, "General (Field)"),
    _S("UET.PHY1096", "General Physics 2", 3, "General (Field)",
       {"subject": "UET.PHY1095"}, [], _req("UET.PHY1095")),
    _S("UET.COM1050", "Computational Thinking", 5, "General (Field)"),
    _S("UET.MAT1052", "Probability and Statistics", 3, "General (Field)",
       {"subject": "UET.MAT1050"}, [], _req("UET.MAT1050")),
    _S("UET.MAT1057", "Discrete Mathematics", 3, "General (Field)",
       {"subject": "UET.MAT1053"}, [], _req("UET.MAT1053")),
    _S("UET.CS1058", "Data Structures and Algorithms", 3, "General (Field)",
       {"subject": "UET.COM1050"}, [], _req("UET.COM1050")),
]

# -- II.1 Foundation (21 credits: 18 compulsory + 3 elective) -----------------
_FOUNDATION = [
    _S("UET.CS2043", "Advanced Programming", 3, "Foundation",
       {"subject": "UET.COM1050"}, [], _req("UET.COM1050")),
    _S("UET.IS2099", "Database", 3, "Foundation"),
    _S("UET.CN2042", "Computer Network", 3, "Foundation",
       {"subject": "UET.COM1050"}, [], _req("UET.COM1050")),
    _S("UET.CS2045", "Software Engineering", 3, "Foundation",
       {"subject": "UET.CS2043"}, [], _req("UET.CS2043")),
    _S("UET.CS2046", "Artificial Intelligence", 3, "Foundation",
       {"subject": "UET.CS1058"}, [], _req("UET.CS1058")),
    _S("UET.IS2100", "Fundamentals of Operating Systems", 3, "Foundation"),
]
_FOUNDATION_ELECTIVE = [
    _S("UET.CE2021", "Computer Architecture", 3, "Foundation",
       {"subject": "UET.COM1050"}, [], _req("UET.COM1050")),
    _S("UET.CS2047", "Compiling Techniques", 3, "Foundation",
       {"subject": "UET.CS1058"}, [], _req("UET.CS1058")),
    _S("UET.CS2048", "Advanced Algorithms and Applications", 3, "Foundation",
       {"subject": "UET.CS1058"}, [], _req("UET.CS1058")),
    _S("UET.MAT2044", "Optimization", 3, "Foundation",
       _and("UET.CS1058", "UET.MAT1057"), [], _req("UET.CS1058", "UET.MAT1057")),
]

# -- II.2 Core (42 credits: 12 compulsory + 30 elective) ----------------------
_CORE = [
    _S("UET.CS3136", "Machine Learning", 3, "Core",
       _and("UET.COM1050", "UET.MAT1052"), [], _req("UET.COM1050", "UET.MAT1052")),
    _S("UET.CS3137", "Introduction to Data Science", 3, "Core",
       _and("UET.COM1050", "UET.MAT1052"), [], _req("UET.COM1050", "UET.MAT1052")),
    _S("UET.CS3138", "Undergraduate Research Project", 3, "Core",
       {"subject": "UET.CS3139"}, [], _req("UET.CS3139")),
    _S("UET.CS3139", "Advanced Topics in Computer Science", 3, "Core"),
]
_CORE_ELECTIVE = [
    _S("UET.AI3056", "Deep Learning", 3, "Core",
       {"subject": "UET.CS3136"}, [], _req("UET.CS3136")),
    _S("UET.AI3059", "Massive Parallel Programming with GPU", 3, "Core",
       {"subject": "UET.CS2043"}, [], _req("UET.CS2043")),
    _S("UET.CS3144", "Scientific Computing for Machine Learning", 3, "Core",
       _and("UET.MAT1051", "UET.MAT1053"), [], _req("UET.MAT1051", "UET.MAT1053")),
    _S("UET.IT3294", "Program Analysis and Testing", 3, "Core",
       {"subject": "UET.CS2043"}, [], _req("UET.CS2043")),
    _S("UET.AI3058", "Reinforcement Learning and Planning", 3, "Core",
       _and("UET.CS3136", "UET.MAT1052"), [], _req("UET.CS3136", "UET.MAT1052")),
    _S("UET.CS3140", "Information Theory", 3, "Core",
       {"subject": "UET.MAT1052"}, [], _req("UET.MAT1052")),
    _S("UET.IT3290", "IT Project Management", 3, "Core",
       {"subject": "UET.CS2045"}, [], _req("UET.CS2045")),
    _S("UET.CS3147", "Graph Theory for Machine Learning", 3, "Core",
       _and("UET.CS1058", "UET.CS3136"), [], _req("UET.CS1058", "UET.CS3136")),
    _S("UET.IT3297", "AI Engineering", 3, "Core",
       _and("UET.CS1058", "UET.CS2045", "UET.CS2046"), [],
       _req("UET.CS1058", "UET.CS2045", "UET.CS2046")),
    _S("UET.CS3142", "Natural Language Processing", 3, "Core",
       {"subject": "UET.CS1058"}, [], _req("UET.CS1058")),
    _S("UET.CS3145", "Deployment and Optimization for Scalable Systems", 3, "Core",
       _and("UET.CN2042", "UET.COM1050"), [], _req("UET.CN2042", "UET.COM1050")),
    _S("UET.AI3067", "Large Language Models and Applications", 3, "Core",
       {"subject": "UET.CS3136"}, [], _req("UET.CS3136")),
    _S("UET.CS3143", "Bioinformatics and Its Applications", 3, "Core",
       {"subject": "UET.CS2043"}, [], _req("UET.CS2043")),
    _S("UET.CS3146", "Special Topics in Computer Science", 3, "Core"),
    _S("UET.CS3148", "Geospatial Data Analysis and Applications", 3, "Core",
       _and("UET.CS3137", "UET.MAT1052"), [], _req("UET.CS3137", "UET.MAT1052")),
    _S("UET.IS3281", "Blockchain and Distributed Ledger Technologies", 3, "Core"),
    _S("UET.IS3276", "Big Data Analytics", 3, "Core",
       _or("UET.CS3136", "UET.IS3278"), [],
       "Prerequisite: UET.CS3136 or UET.IS3278."),
    _S("UET.IS3286", "Business Analytics", 3, "Core",
       _and("UET.IS2099", "UET.MAT1052"), [], _req("UET.IS2099", "UET.MAT1052")),
    _S("UET.CS3152", "User Interface and User Experience Design", 3, "Core",
       {"subject": "UET.CS2045"}, [], _req("UET.CS2045")),
    _S("UET.CS3153", "Multimedia Communications", 3, "Core",
       _and("UET.COM1050", "UET.CS3140"), [], _req("UET.COM1050", "UET.CS3140")),
    _S("UET.IT3289", "Cross-platform Application Development", 3, "Core",
       {"subject": "UET.CS2043"}, [], _req("UET.CS2043")),
    _S("UET.IT3291", "System Analysis and Design", 3, "Core",
       {"subject": "UET.CS2045"}, [], _req("UET.CS2045")),
    _S("UET.CE2020", "Systems Programming", 3, "Core",
       {"subject": "UET.COM1050"}, [], _req("UET.COM1050")),
    _S("UET.CS3141", "Computer Graphics", 3, "Core",
       {"subject": "UET.CS2043"}, [], _req("UET.CS2043")),
    _S("UET.CS3150", "Image Processing and Computer Vision", 3, "Core",
       {"subject": "UET.CS1058"}, [], _req("UET.CS1058")),
    _S("UET.CS3151", "Human-Computer Interaction", 3, "Core",
       {"subject": "UET.CS2045"}, [], _req("UET.CS2045")),
    _S("UET.IT3292", "Designing Large-scale Software Systems", 3, "Core",
       {"subject": "UET.IT3291"}, [], _req("UET.IT3291")),
    _S("UET.CS3149", "Introduction to Cognitive Intelligence", 3, "Core"),
]

# -- III Supplementary (3 elective credits from a school-defined list) --------
# The official III.2 elective list is not part of the provided curriculum, so a
# single placeholder represents "one supplementary elective".
_SUPPLEMENTARY = [
    _S("UET.SUP001", "Supplementary Elective (school-defined list)", 3,
       "Supplementary"),
]

# -- IV Internship and graduation (13 credits) -------------------------------
# Thesis track. (The capstone alternative — UET.CS4024 + 6 core-elective credits —
# is not modelled; thesis is treated as the required graduation path.)
_GRADUATION = [
    _S("UET.CS4022", "Industrial Internship", 3, "Graduation",
       {"subject": "UET.CS3138"}, [], _req("UET.CS3138")),
    _S("UET.CS4023", "Graduation Thesis", 10, "Graduation",
       {"subject": "UET.CS3138"}, [], _req("UET.CS3138")),
]

_SUBJECTS = (_GEN_VNU + _GEN_FIELD + _FOUNDATION + _FOUNDATION_ELECTIVE
             + _CORE + _CORE_ELECTIVE + _SUPPLEMENTARY + _GRADUATION)

SUBJECTS = {s.code: s for s in _SUBJECTS}

# Compulsory courses (must be completed). Electives are satisfied via the category
# credit minimums below, with the planner selecting which ones to take.
REQUIRED = (
    [s.code for s in _GEN_VNU]
    + [s.code for s in _GEN_FIELD]
    + [s.code for s in _FOUNDATION]
    + [s.code for s in _CORE]
    + [s.code for s in _GRADUATION]
)

# Credit minimum per block. Compulsory courses already supply part of each; the
# rest must come from electives in the same category.
CATEGORY_MIN = {
    "General (VNU)": 21,
    "General (Field)": 35,
    "Foundation": 21,    # 18 compulsory + 3 elective
    "Core": 42,          # 12 compulsory + 30 elective
    "Supplementary": 3,
    "Graduation": 13,
}

TOTAL_CREDITS = 135

PROGRAM = Program(
    name="B.Sc. Computer Science (UET-VNU)",
    subjects=SUBJECTS,
    category_min_credits=CATEGORY_MIN,
    required_codes=REQUIRED,
    total_credits_required=TOTAL_CREDITS,
    num_terms=8,            # nominal 4-year, on-time graduation target
    max_credits_per_term=18,
    min_credits_per_term=12,
    max_terms=12,           # allow up to 6 years so delays can extend graduation
)
