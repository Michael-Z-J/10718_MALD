"""prompts and json schemas of the llm systems.

all systems get the same task header, rubric text and output schemas, so a
difference in results comes from the pipeline and not from the output format.
the prompt texts are part of the llm cache key: a change to them invalidates
the cache.
"""
AXES = ["soundness", "presentation", "contribution"]
OVERALL = [1, 3, 5, 6, 8, 10]

HEADER = """You help ML researchers before they submit a paper to ICLR. The authors have about four weeks before the deadline. They will revise the paper and then run you again on the full revised paper.

Rubric (ICLR scale):
- soundness (1-4): are the claims correct and supported by the method, proofs and experiments?
- presentation (1-4): is the paper clear, complete and well organized?
- contribution (1-4): are the ideas and results novel and significant?
- overall (1, 3, 5, 6, 8, 10): 1 strong reject, 3 reject, 5 marginally below the threshold, 6 marginally above the threshold, 8 accept, 10 strong accept.

Each weakness must name one specific problem of this paper, point to where it is (section, table, claim), and say what change would fix it. "effort" is the time the authors need for the fix: "days", "weeks" or "months"."""

ISSUE = {
    "type": "object",
    "properties": {
        "issue": {"type": "string"},
        "axis": {"enum": AXES},
        "fix": {"type": "string"},
        "effort": {"enum": ["days", "weeks", "months"]},
    },
    "required": ["issue", "axis", "fix", "effort"],
}
SCORES = {
    "type": "object",
    "properties": {**{a: {"enum": [1, 2, 3, 4]} for a in AXES}, "overall": {"enum": OVERALL}},
    "required": AXES + ["overall"],
}
REVIEW = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "weaknesses": {"type": "array", "items": ISSUE, "minItems": 2, "maxItems": 10},
        "questions": {"type": "object", "properties": {a: {"type": "array", "items": {"type": "string"}, "maxItems": 3} for a in AXES},
                      "required": AXES},
        "scores": SCORES,
        "confidence": {"enum": [1, 2, 3, 4, 5]},
    },
    "required": ["summary", "strengths", "weaknesses", "questions", "scores", "confidence"],
}
META_ISSUE = {
    "type": "object",
    "properties": {
        "issue": {"type": "string"},
        "axis": {"enum": AXES},
        "raised_by": {"type": "array", "items": {"type": "integer"}},
        "relevant": {"type": "boolean"},
        "doable": {"type": "boolean"},
    },
    "required": ["issue", "axis", "raised_by", "relevant", "doable"],
}
META = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": META_ISSUE, "minItems": 1, "maxItems": 20},
        "scores": SCORES,
        "decision": {"enum": ["accept", "reject"]},
    },
    "required": ["summary", "issues", "scores", "decision"],
}
PANEL = {
    "type": "object",
    "properties": {"reviews": {"type": "array", "items": REVIEW, "minItems": 6, "maxItems": 6}, "meta": META},
    "required": ["reviews", "meta"],
}

META_RULES = """The meta-review lists the issues in order of importance for the decision, most important first. Merge issues that several reviewers share; "raised_by" lists the reviewer numbers (1-based). Set "relevant" to true if fixing the issue would change what reviewers think of the paper, and false if it is minor or a reviewer is wrong. Set "doable" to true if the authors can fix it in the four weeks before the deadline. Do not leave out any issue that a reviewer raised; mark weak ones as not relevant instead."""

# ours (b1): one prompt writes the panel and the meta-review in one generation.
B1 = """Below is a paper draft. Simulate the ICLR review process for it in one answer. First write six independent reviews by six different reviewers, as real ICLR reviewers with different backgrounds and strictness would write them. Then write the area chair's meta-review of those six reviews.

""" + META_RULES + """

Return only JSON.

PAPER:
{paper}"""

# ml baseline (b1old): top-5 weaknesses prompt of the earlier baseline.
B1_OLD = """Below is the full text of a paper submitted to ICLR. Acting as a peer reviewer, list the five most important weaknesses of this paper. For each, give a severity from 1 (minor) to 5 (critical). Return only JSON, with no other text.

{paper}"""
B1_OLD_SCHEMA = {
    "type": "object",
    "properties": {"weaknesses": {"type": "array", "minItems": 5, "maxItems": 5, "items": {
        "type": "object", "properties": {"weakness": {"type": "string"}, "severity": {"enum": [1, 2, 3, 4, 5]}},
        "required": ["weakness", "severity"]}}},
    "required": ["weaknesses"],
}

# ours, agentic (m): one reviewer of the panel.
REVIEWER = """Below is a paper draft. You are reviewer {i} of a panel of six. Write your own independent review.

PAPER:
{paper}

{persona}

Return only JSON."""

PERSONA_NONE = "Review the paper as a typical ICLR reviewer."

PERSONA_UNCOND = """REFERENCE REVIEW. A real reviewer wrote the review below for a related paper at an earlier ICLR and gave it overall {overall}. It shows you how strict this reviewer is, what they focus on and how detailed they are. Review the new paper as this reviewer would. Do not copy its points; the papers are different.

{review}"""

PERSONA_REBUTTAL = """REFERENCE REVIEWER. A real reviewer wrote the review below for a related paper at an earlier ICLR. After the authors' response, the same reviewer wrote the follow-up shown below it, which says that the authors' changes addressed some concerns. Review the new paper as this reviewer would: raise the concerns that matter for your score, and for each one say what change or evidence would make you raise your score. Prefer concerns that the authors can fix with work before the deadline. Do not copy its points; the papers are different.

REVIEW:
{review}

FOLLOW-UP AFTER THE AUTHORS' RESPONSE:
{followup}"""

META_PROMPT = """You are the area chair. Six reviewers reviewed the paper below. The table groups their weaknesses into issues. For each issue it gives the reviewers who raised it, the axis, how much effort the reviewers think the fix needs, and one or more statements of the issue.{modes}

PAPER TITLE: {title}
ABSTRACT: {abstract}

ISSUES (sorted by number of reviewers{deficit}):
{table}
{scores}
Write the meta-review. Include every issue in the table. """ + META_RULES + """

Return only JSON."""
