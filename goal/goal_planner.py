# goal/goal_planner.py
from typing import List, Dict, Any, Tuple
from spider_core.llm.openai_gpt_client import OpenAIGPTClient

# ---------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------

GOAL_CHUNK_SYSTEM = (
    "You are a goal-driven web research planner. "
    "Given a user GOAL and a PAGE CHUNK, you will: "
    "1) estimate if the GOAL is fully answered by this page content (0-1), "
    "2) extract a concise answer delta (new facts that progress the goal), "
    "3) propose next links (subset of candidates) most likely to progress the goal.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "{"
    '  \"goal_satisfaction_estimate\": float (0..1),'
    '  \"answer_delta\": \"short string\",'
    '  \"next_link_scores\": [{\"href\": \"...\", \"score\": float (0..1)}]'
    "}"
)

GOAL_CONTEXT_SYSTEM = (
    "You are an evaluator for a goal-oriented web crawler.\n"
    "Your job is to decide whether the user's GOAL has been answered by the given CONTEXT.\n\n"
    "Rules:\n"
    "- Be strict but pragmatic: only mark found=true if the answer is explicitly in CONTEXT.\n"
    "- Always extract the literal answer text (e.g., street address, email, name, etc.).\n"
    "- Never say 'provided above' or 'in the context'; quote the exact answer.\n"
    "- 'confidence' is a float (0..1) expressing how sure you are that this is the correct answer.\n"
    "- 'answer' must be short and direct, extracted verbatim from CONTEXT.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "{"
    '  \"found\": bool,'
    '  \"confidence\": float (0..1),'
    '  \"answer\": \"short string\"'
    "}"
)

# New prompt: self-grade sanity check
GOAL_GRADE_SYSTEM = (
    "You are a strict grader verifying whether an extracted ANSWER correctly satisfies a GOAL.\n"
    "Evaluate correctness, completeness, and specificity.\n\n"
    "Return ONLY JSON with keys:\n"
    "{"
    '  \"valid\": bool,'
    '  \"confidence\": float (0..1),'
    '  \"reason\": \"short justification\"'
    "}"
)

LINK_PLANNER_SYSTEM = (
    "You are a navigation planner for a goal-oriented web crawler.\n"
    "Given a GOAL, CURRENT PAGE URL, PAGE SNIPPET, current CONFIDENCE, and a list of CANDIDATE LINKS, "
    "assign each link a score 0.0–1.0 indicating how promising it is for achieving the GOAL.\n\n"
    "Handle arbitrary goals (contacts, products, biographies, etc.). Prefer links semantically related to the GOAL.\n\n"
    "Return ONLY valid JSON:\n"
    "{"
    '  \"link_scores\": ['
    '    {\"href\": \"https://example.com/path\", \"score\": 0.0},'
    '    ...'
    "  ]"
    "}"
)


def build_chunk_prompt(goal: str, chunk_text: str, link_candidates: List[Dict[str, str]]) -> str:
    return (
        f"GOAL:\n{goal}\n\n"
        f"PAGE CHUNK (truncated):\n{chunk_text[:4000]}\n\n"
        f"LINK CANDIDATES (href + text):\n"
        f"{[{'href': l['href'], 'text': l.get('text', '')[:200]} for l in link_candidates]}\n\n"
        "Return JSON as described."
    )


class GoalPlanner:
    """High-level planner for goal-oriented crawling with RAG evaluation and answer validation."""

    def __init__(self, llm: OpenAIGPTClient):
        self.llm = llm

    # --- Context-level goal evaluation (RAG) ---
    async def evaluate_context(self, goal: str, context_text: str) -> Tuple[bool, float, str]:
        if not context_text.strip():
            return False, 0.0, ""
        prompt = f"GOAL:\n{goal}\n\nCONTEXT (truncated):\n{context_text[:8000]}"
        out = await self.llm.complete_json(GOAL_CONTEXT_SYSTEM, prompt)
        found = bool(out.get("found", False))
        confidence = float(out.get("confidence", 0.0))
        answer = out.get("answer", "").strip()
        return found, confidence, answer

    # --- Sanity check on extracted answer ---
    async def grade_answer(self, goal: str, answer: str) -> Tuple[bool, float, str]:
        """Verify that the extracted answer actually satisfies the goal."""
        if not answer.strip():
            return False, 0.0, "Empty answer."
        prompt = f"GOAL:\n{goal}\n\nANSWER:\n{answer}"
        out = await self.llm.complete_json(GOAL_GRADE_SYSTEM, prompt)
        valid = bool(out.get("valid", False))
        confidence = float(out.get("confidence", 0.0))
        reason = out.get("reason", "").strip()
        return valid, confidence, reason

    # --- Goal-aware link scoring ---
    async def score_links(
        self,
        goal: str,
        page_url: str,
        page_snippet: str,
        link_candidates: List[Dict[str, str]],
        current_confidence: float,
        max_links: int = 50,
    ) -> Dict[str, float]:
        if not link_candidates:
            return {}
        trimmed = link_candidates[:max_links]
        page_snippet = (page_snippet or "")[:1500]
        user_prompt = (
            f"GOAL:\n{goal}\n\nCURRENT CONFIDENCE:\n{current_confidence}\n\n"
            f"PAGE URL:\n{page_url}\n\nPAGE SNIPPET:\n{page_snippet}\n\n"
            f"CANDIDATE LINKS:\n"
            f"{[{'href': l.get('href'), 'text': (l.get('text') or '')[:200]} for l in trimmed]}"
        )
        out = await self.llm.complete_json(LINK_PLANNER_SYSTEM, user_prompt)
        scores = {}
        for item in out.get("link_scores", []) or []:
            href = item.get("href")
            if not href:
                continue
            try:
                s = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                s = 0.0
            scores[href] = s
        return scores
