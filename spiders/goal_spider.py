import asyncio, heapq, logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from spider_core.spiders.basic_spider import BasicSpider, maybe_await
from spider_core.goal.goal_planner import GoalPlanner
from spider_core.storage.db import DB
from spider_core.llm.embeddings_client import EmbeddingsClient, OpenAIEmbeddings
from spider_core.base.page_result import PageResult
from spider_core.base.link_metadata import LinkMetadata

logger = logging.getLogger("goal_spider")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[GoalSpider] %(levelname)s %(message)s"))
    logger.addHandler(ch)

class GoalOrientedSpider(BasicSpider):
    """
    Goal-driven crawler that:
     - Starts from a seed URL
     - Iteratively fetches pages based on goal-conditioned relevance
     - Aggregates an answer and stops when confidence >= threshold or budget exhausted
     - Stores pages/chunks in SQLite and vectorizes chunks for RAG
    """
    def __init__(self,
                 browser_client,
                 relevance_ranker,
                 chunker,
                 planner: GoalPlanner,
                 db: DB,
                 embedder: Optional[EmbeddingsClient] = None,
                 embed_model: str = "text-embedding-3-small",
                 stop_threshold: float = 0.85,
                 max_pages: int = 20):
        super().__init__(browser_client, relevance_ranker, chunker)
        self.planner = planner
        self.db = db
        self.embedder = embedder or OpenAIEmbeddings(model=embed_model)
        self.embed_model = embed_model
        self.stop_threshold = stop_threshold
        self.max_pages = max_pages

    async def fetch_goal(self, seed_url: str, goal: str) -> Dict:
        """
        Returns dict with final answer, confidence, visited_count, and trace.
        """
        # frontier: max-heap by priority (negated for heapq)
        frontier: List[Tuple[float, str, Optional[str]]] = []
        seen = set()
        answer_parts: List[str] = []
        final_conf = 0.0
        visited = 0
        trace = []

        # seed
        heapq.heappush(frontier, (-1.0, seed_url, None))  # (priority, url, from_url)

        while frontier and visited < self.max_pages and final_conf < self.stop_threshold:
            priority, url, parent = heapq.heappop(frontier)
            if url in seen:
                continue
            seen.add(url)

            # fetch page (no LLM scoring yet; BasicSpider flow)
            page: PageResult = await self._fetch_without_llm(url)

            # persist page + links + chunks
            title = None
            try:
                # naive title grab from first chunk (can be improved)
                first_text = page.page_chunks[0]["text"] if page.page_chunks else ""
                title = (first_text.split("\n", 1)[0] or "").strip()[:200] or None
            except Exception:
                pass

            self.db.upsert_page(url, page.canonical, page.status, title, "\n\n".join([c["text"] for c in page.page_chunks or []]))
            self.db.upsert_links(url, [l.__dict__ for l in page.links])
            self.db.upsert_chunks(url, page.page_chunks or [])
            self.db.log(url, "fetched", reason=f"priority={-priority}")

            # embed chunks
            texts = [c["text"] for c in page.page_chunks or []]
            if texts:
                vecs = self.embedder.embed(texts)
                for c, v in zip(page.page_chunks, vecs):
                    self.db.upsert_embedding(url, c["chunk_id"], v, self.embed_model, len(v))

            # goal-conditioned evaluation over chunks
            goal_conf_this_page = 0.0
            link_scores_accum: Dict[str, float] = {}
            for chunk in (page.page_chunks or []):
                est, delta, scored = await self.planner.evaluate_chunk(
                    goal, chunk["text"],
                    [{"href": l.href, "text": l.text or ""} for l in page.links]
                )
                goal_conf_this_page = max(goal_conf_this_page, est)
                if delta:
                    answer_parts.append(delta)
                # merge scores
                for href, s in scored.items():
                    link_scores_accum[href] = max(link_scores_accum.get(href, 0.0), s)

            final_conf = max(final_conf, goal_conf_this_page)
            trace.append({"url": url, "page_conf": goal_conf_this_page, "picked": sorted(link_scores_accum.items(), key=lambda x: x[1], reverse=True)[:5]})
            visited += 1

            # rank outgoing links by goal-conditioned scores; push to frontier
            for l in page.links:
                score = float(link_scores_accum.get(l.href, 0.0))
                # if planner didn't score it, fall back to LLM link relevance (if present)
                if score == 0.0:
                    score = float(getattr(l, "llm_score", 0.0))
                if l.href not in seen and score > 0.0:
                    heapq.heappush(frontier, (-(score + 1e-6), l.href, url))
                    self.db.set_final_link_score(url, l.href, score)

        return {
            "goal": goal,
            "confidence": final_conf,
            "visited_count": visited,
            "answer": "\n".join(answer_parts).strip(),
            "trace": trace,
        }
