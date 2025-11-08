import asyncio
import logging
from datetime import datetime
from typing import List
from spider_core.base.spider import Spider
from spider_core.base.page_result import PageResult
from spider_core.browser.browser_client import BrowserClient
from spider_core.extractors.deterministic_extractor import DeterministicLinkExtractor
from spider_core.core_utils.chunking import TextChunker
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.base.link_metadata import LinkMetadata

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
    render → extract → chunk → (optional) LLM scoring
    """

    def __init__(self, browser_client: BrowserClient, relevance_ranker: RelevanceRanker, chunker: TextChunker):
        self.browser_client = browser_client
        self.relevance_ranker = relevance_ranker
        self.chunker = chunker

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
        logger.info("Scoring links with LLM...")
        rr = self.relevance_ranker
        method = next((getattr(rr, n) for n in ("score_links", "rank_links", "rank", "score") if hasattr(rr, n)), None)
        if not method:
            return
        res = method(page_result.links, page_result.page_chunks)
        await maybe_await(res)
        logger.info("LLM scoring completed.")

    def summarize_result(self, page_result: PageResult) -> str:
        return (
            f"URL: {page_result.url}\n"
            f"Status: {page_result.status}\n"
            f"Links found: {len(page_result.links)}\n"
            f"Text chunks: {len(page_result.page_chunks)}\n"
        )
