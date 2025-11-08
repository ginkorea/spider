import asyncio
import logging
from spider_core.spiders.basic_spider import BasicSpider
from spider_core.spiders.stealth.vpn_manager import VPNManager, VPNError
from spider_core.spiders.stealth.stealth_config import (
    DEFAULT_REGION,
    DEFAULT_VPN_PROVIDER,
    REQUIRE_VPN_DEFAULT,
    DISCONNECT_BEFORE_LLM,
    RECONNECT_AFTER_LLM,
    OBFUSCATE_BY_DEFAULT,
    PROTOCOL_DEFAULT,
)

logger = logging.getLogger("StealthSpider")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[StealthSpider] %(message)s"))
    logger.addHandler(ch)


class StealthSpider(BasicSpider):
    """Adds VPN routing around BasicSpider.fetch()"""

    def __init__(
        self,
        browser_client,
        relevance_ranker,
        chunker,
        vpn_provider=DEFAULT_VPN_PROVIDER,
        region=DEFAULT_REGION,
        require_vpn=REQUIRE_VPN_DEFAULT,
        disconnect_before_llm=DISCONNECT_BEFORE_LLM,
        reconnect_after_llm=RECONNECT_AFTER_LLM,
        obfuscate=OBFUSCATE_BY_DEFAULT,
        protocol=PROTOCOL_DEFAULT,
    ):
        super().__init__(browser_client, relevance_ranker, chunker)
        self.vpn = VPNManager(provider=vpn_provider)
        self.vpn_provider = vpn_provider
        self.region = region
        self.require_vpn = require_vpn
        self.disconnect_before_llm = disconnect_before_llm
        self.reconnect_after_llm = reconnect_after_llm
        self.obfuscate = obfuscate
        self.protocol = protocol

    async def fetch(self, url: str):
        """Full VPN-controlled fetch."""
        if self.require_vpn:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.vpn.connect(self.region, obfuscate=self.obfuscate, protocol=self.protocol)
                )
            except Exception as e:
                raise VPNError(f"VPN connection failed: {e}")
        logger.info(f"Fetching under VPN: {url}")
        page = await self._fetch_without_llm(url)
        if self.disconnect_before_llm:
            await asyncio.get_event_loop().run_in_executor(None, self.vpn.disconnect)
        await self._score_links_with_llm(page)
        if self.reconnect_after_llm:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.vpn.connect(self.region, obfuscate=self.obfuscate, protocol=self.protocol)
            )
        return page
