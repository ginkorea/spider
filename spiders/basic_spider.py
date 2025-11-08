import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import List
from spider_core.base.spider import Spider
from spider_core.base.page_result import PageResult
from spider_core.browser.browser_client import BrowserClient
from spider_core.extractors.deterministic_extractor import DeterministicLinkExtractor
from spider_core.core_utils.chunking import TextChunker
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.base.link_metadata import LinkMetadata
from spider_core.llm.embeddings_client import OpenAIEmbeddings, cosine_sim

logger = logging.getLogger("BasicSpider")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[BasicSpider] %(message)s"))
    logger.addHandler(ch)


async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class BasicSpider(Spider):
    """
    Handles the deterministic pipeline:
      render → extract → chunk → (optional) link ranking

    Features:
    - Async rendering and chunking
    - Hybrid embedding prefilter + async LLM link scoring (goal mode)
    - Lightweight lexical ranking (generic mode)
    """

    def __init__(self, browser_client: BrowserClient, relevance_ranker: RelevanceRanker, chunker: TextChunker):
        self.browser_client = browser_client
        self.relevance_ranker = relevance_ranker
        self.chunker = chunker
        self.embedder = None
        self.llm = getattr(relevance_ranker, "llm_client", None)
        self.semaphore = asyncio.Semaphore(8)  # limit concurrent LLM calls

    async def fetch(self, url: str) -> PageResult:
        logger.info(f"Starting fetch pipeline for: {url}")
        page_result = await self._fetch_without_llm(url)

        if page_result.page_chunks and self.relevance_ranker:
            try:
                await self._score_links_with_llm(page_result)
            except Exception as e:
                logger.warning(f"LLM scoring failed: {e}")

        if not page_result.canonical:
            canonical = next((l.href for l in page_result.links if "canonical" in (l.rel or [])), page_result.url)
            page_result.canonical = canonical

        logger.info(f"Finished fetch pipeline for: {url} (links={len(page_result.links)})")
        return page_result

    async def _fetch_without_llm(self, url: str) -> PageResult:
        html, visible_text, status = await self.browser_client.render(url)
        links: List[LinkMetadata] = []
        try:
            links = DeterministicLinkExtractor.extract(html, url)
        except Exception as e:
            logger.warning(f"Deterministic extractor failed: {e}")
        chunks = self.chunker.chunk_text(visible_text)
        return PageResult(
            url=url,
            fetched_at=datetime.utcnow(),
            status=status,
            canonical=None,
            links=links,
            llm_summary=None,
            page_chunks=chunks,
        )

    async def _score_links_with_llm(self, page_result: PageResult):
        """
        Hybrid fast mode:
        - If a goal is set → embedding prefilter + async LLM scoring
        - If no goal → cheap lexical heuristic ranking (no LLM calls)
        """
        links = page_result.links
        if not links:
            return

        goal_text = getattr(self, "current_goal", None)
        goal_mode = bool(goal_text)

        # --- 🧩 1️⃣ No-goal mode: lightweight lexical ranking ---
        if not goal_mode:
            logger.info("No goal provided — using lightweight lexical ranking.")
            keywords = {"contact", "about", "privacy", "terms", "help", "faq", "support", "company"}
            for link in links:
                text = (link.text or link.href or "").lower()
                score = 0.0
                for kw in keywords:
                    if kw in text:
                        score += 0.2
                if len(text) < 40:
                    score += 0.05  # short nav links are often structural
                link.llm_score = min(score, 1.0)
            return

        # --- ⚡ 2️⃣ Goal mode: embedding prefilter + parallel LLM scoring ---
        logger.info("Scoring links with hybrid async LLM pipeline...")
        try:
            if self.embedder is None:
                self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
            goal_vec = self.embedder.embed([goal_text])[0]
            link_texts = [(l.href, (l.text or l.href or "")[:200]) for l in links]
            link_vecs = self.embedder.embed([t for _, t in link_texts])

            sims = [
                cosine_sim(np.array(goal_vec), np.array(vec))
                for vec in link_vecs
            ]
            ranked_links = sorted(zip(links, sims), key=lambda x: x[1], reverse=True)
            top_links = [l for l, _ in ranked_links[:15]]
        except Exception as e:
            logger.warning(f"Embedding prefilter failed: {e}")
            top_links = links[:10]

        if not hasattr(self, "llm") or self.llm is None:
            logger.warning("No LLM client attached; skipping scoring.")
            return

        async def score_one_link(link: LinkMetadata):
            async with self.semaphore:
                try:
                    prompt = (
                        f"GOAL:\n{goal_text}\n\n"
                        f"LINK:\n{link.href}\n"
                        f"ANCHOR TEXT:\n{link.text or '(none)'}\n"
                        f"CONTEXT SAMPLE:\n"
                        f"{page_result.page_chunks[0]['text'][:800] if page_result.page_chunks else ''}\n\n"
                        "Rate how relevant this link is for achieving the goal.\n"
                        "Return JSON: {\"score\": float (0..1)}"
                    )
                    result = await self.llm.complete_json(
                        "You are a relevance evaluator for a web crawler.", prompt
                    )
                    link.llm_score = float(result.get("score", 0.0))
                except Exception as e:
                    logger.debug(f"LLM failed on {link.href}: {e}")
                    link.llm_score = 0.0
                return link

        # Run LLM scoring concurrently
        scored_links = await asyncio.gather(*(score_one_link(l) for l in top_links))
        scored_links.sort(key=lambda l: getattr(l, "llm_score", 0.0), reverse=True)
        logger.info(f"Parallel LLM scoring completed ({len(scored_links)} links).")

        # Merge scores back into full link set
        scored_map = {l.href: getattr(l, "llm_score", 0.0) for l in scored_links}
        for link in page_result.links:
            link.llm_score = scored_map.get(link.href, getattr(link, "llm_score", 0.0))

    def summarize_result(self, page_result: PageResult) -> str:
        return (
            f"URL: {page_result.url}\n"
            f"Status: {page_result.status}\n"
            f"Links found: {len(page_result.links)}\n"
            f"Text chunks: {len(page_result.page_chunks)}\n"
        )
