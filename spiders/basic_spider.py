import asyncio
from datetime import datetime
from spider_core.base.spider import Spider
from spider_core.base.page_result import PageResult
from spider_core.browser.browser_client import BrowserClient
from spider_core.extractors.deterministic_extractor import DeterministicLinkExtractor
from spider_core.core_utils.chunking import TextChunker
from spider_core.llm.relevance_ranker import RelevanceRanker


class BasicSpider(Spider):
    """
    A full pipeline spider that:
    1. Renders page
    2. Extracts links
    3. Chunks visible text
    4. Scores links using LLM
    """

    def __init__(
        self,
        browser_client: BrowserClient,
        relevance_ranker: RelevanceRanker,
        chunker: TextChunker,
    ):
        self.browser_client = browser_client
        self.relevance_ranker = relevance_ranker
        self.chunker = chunker

    async def fetch(self, url: str) -> PageResult:
        html, visible_text, status = await self.browser_client.render(url)

        links = DeterministicLinkExtractor.extract(html, url)

        chunks = self.chunker.chunk_text(visible_text)

        if chunks:
            links = await self.relevance_ranker.score_links(links, chunks)

        canonical = next((l.href for l in links if "canonical" in l.rel), url)

        return PageResult(
            url=url,
            fetched_at=datetime.utcnow(),
            status=status,
            canonical=canonical,
            links=links,
            llm_summary=None,
            page_chunks=chunks,
        )
