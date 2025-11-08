import heapq
import logging
from typing import Dict, List, Optional, Set, Tuple
from spider_core.spiders.basic_spider import BasicSpider
from spider_core.base.page_result import PageResult
from spider_core.storage.db import DB
from spider_core.goal.goal_planner import GoalPlanner
from spider_core.llm.embeddings_client import EmbeddingsClient, OpenAIEmbeddings

logger = logging.getLogger("GoalSpider")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[GoalSpider] %(message)s"))
    logger.addHandler(ch)


class GoalSpider(BasicSpider):
    """RAG-augmented spider that reasons toward an arbitrary goal with answer grading."""

    def __init__(
        self,
        browser_client,
        relevance_ranker,
        chunker,
        llm_client,
        db: Optional[DB] = None,
        planner: Optional[GoalPlanner] = None,
        embedder: Optional[EmbeddingsClient] = None,
        embed_model: str = "text-embedding-3-small",
        max_depth: int = 3,
        stop_threshold: float = 0.9,
        max_pages: int = 25,
        retrieval_top_k: int = 8,
    ):
        super().__init__(browser_client, relevance_ranker, chunker)
        self.llm = llm_client
        self.db = db or DB()
        self.planner = planner or GoalPlanner(self.llm)
        self.embed_model = embed_model
        self.embedder = embedder or OpenAIEmbeddings(model=embed_model)
        self.max_depth = max_depth
        self.stop_threshold = stop_threshold
        self.max_pages = max_pages
        self.retrieval_top_k = retrieval_top_k
        self.visited: Set[str] = set()
        self._stability_counter = 0
        self.current_goal = None  # propagate to BasicSpider

    async def _store_page(self, page: PageResult):
        """Store page text, chunks, and embeddings for later retrieval."""
        try:
            visible_text = "\n".join(c["text"] for c in page.page_chunks)
            self.db.upsert_page(
                page.url, page.canonical, page.status, None, visible_text
            )
            self.db.upsert_links(page.url, [l.__dict__ for l in page.links])
            self.db.upsert_chunks(page.url, page.page_chunks)

            # Embed chunks for RAG
            for chunk in page.page_chunks:
                vec = self.embedder.embed([chunk["text"]])[0]
                self.db.upsert_embedding(
                    page.url,
                    chunk["chunk_id"],
                    vec,
                    self.embed_model,
                    len(vec),
                )
        except Exception as e:
            logger.warning(f"[GoalSpider] Failed to store page {page.url}: {e}")

    async def fetch(self, url: str) -> PageResult:
        """Goal-aware fetch that ensures goal context propagates to BasicSpider."""
        self.current_goal = getattr(self, "current_goal", None)
        logger.info(f"[GoalSpider] Delegating fetch for: {url}")
        return await super().fetch(url)

    async def run_goal(self, start_url: str, goal: str) -> Dict:
        self.current_goal = goal
        heap: List[Tuple[float, str, int]] = [(-1.0, start_url, 0)]
        best_answer, best_conf = "", 0.0

        while heap and len(self.visited) < self.max_pages:
            found, conf, ans = await self._evaluate_memory(goal)
            if conf > best_conf:
                best_conf, best_answer = conf, ans

            # ✅ Sanity check / self-grade
            if found and ans:
                valid, grade_conf, reason = await self.planner.grade_answer(goal, ans)
                conf = min(conf, grade_conf)
                if valid:
                    self._stability_counter += 1
                    logger.info(f"[Sanity] Answer validated ({grade_conf:.2f}): {reason}")
                else:
                    self._stability_counter = 0
                    logger.info(f"[Sanity] Answer rejected ({grade_conf:.2f}): {reason}")

            # Stop condition
            if found and conf >= self.stop_threshold and self._stability_counter >= 2:
                logger.info(f"✅ Goal reached with confidence={conf:.2f} after {len(self.visited)} pages")
                break

            _, url, depth = heapq.heappop(heap)
            if url in self.visited or depth > self.max_depth:
                continue
            self.visited.add(url)

            logger.info(f"[Depth {depth}] Crawling {url}")
            page = await self.fetch(url)
            await self._store_page(page)

            # Goal-aware link selection
            scores = await self._score_links_for_goal(goal, page, best_conf)
            for l in sorted(page.links, key=lambda x: scores.get(x.href, 0), reverse=True)[:5]:
                if l.href not in self.visited:
                    heapq.heappush(heap, (-scores.get(l.href, 0), l.href, depth + 1))

        return {
            "goal": goal,
            "found": bool(best_answer.strip()),
            "confidence": best_conf,
            "answer": best_answer.strip(),
            "visited_pages": len(self.visited),
        }

    async def _evaluate_memory(self, goal: str):
        """RAG retrieval + context evaluation."""
        if not self.db.has_embeddings(self.embed_model):
            return False, 0.0, ""
        q_vec = self.embedder.embed([goal])[0]
        retrieved = self.db.similarity_search(q_vec, top_k=self.retrieval_top_k, model=self.embed_model)
        if not retrieved:
            return False, 0.0, ""
        context_text = "\n\n".join(r["text"] for r in retrieved)
        return await self.planner.evaluate_context(goal, context_text)

    async def _score_links_for_goal(self, goal: str, page: PageResult, current_conf: float) -> Dict[str, float]:
        """LLM-guided scoring of links based on page snippet."""
        try:
            page_snippet = (
                page.page_chunks[0]["text"][:1200] if page.page_chunks else ""
            )
            link_dicts = [
                {"href": l.href, "text": l.text or ""} for l in page.links
            ]
            return await self.planner.score_links(
                goal, page.url, page_snippet, link_dicts, current_conf
            )
        except Exception as e:
            logger.warning(f"[GoalSpider] Link scoring failed: {e}")
            return {}
