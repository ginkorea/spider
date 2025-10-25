# spiders/stealth/stealth_spider.py

from spider_core.spiders.basic_spider import BasicSpider
from spider_core.spiders.stealth.vpn_manager import NordVPNManager
from spider_core.spiders.stealth.stealth_config import (
    DEFAULT_REGION,
    DEFAULT_VPN_PROVIDER,
    REQUIRE_VPN_DEFAULT,
)


class StealthSpider(BasicSpider):
    """
    A stealth-enabled spider that ensures traffic is routed through a
    secure anonymization layer (e.g., NordVPN).
    """

    def __init__(
        self,
        browser_client,
        relevance_ranker,
        chunker,
        vpn_provider: str = DEFAULT_VPN_PROVIDER,
        region: str = DEFAULT_REGION,
        require_vpn: bool = REQUIRE_VPN_DEFAULT,
        tor_enabled: bool = False,
    ):
        super().__init__(browser_client, relevance_ranker, chunker)
        self.vpn_provider = vpn_provider
        self.region = region
        self.require_vpn = require_vpn
        self.tor_enabled = tor_enabled

        if self.vpn_provider == "nordvpn":
            self.vpn_manager = NordVPNManager()
        else:
            raise NotImplementedError(f"VPN provider '{self.vpn_provider}' is not supported yet.")

    async def _ensure_stealth(self):
        """Ensure VPN is connected and aligned with requested region."""
        if self.vpn_provider == "nordvpn":
            try:
                if not self.vpn_manager.is_in_region(self.region):
                    print(f"[StealthSpider] Attempting VPN connection to: {self.region}")
                    self.vpn_manager.ensure_region(self.region)
                    print(f"[StealthSpider] VPN connected to {self.region}")
            except Exception as e:
                if self.require_vpn:
                    raise RuntimeError(f"[StealthSpider] VPN enforcement failed: {e}")
                else:
                    print(f"[StealthSpider WARNING] VPN not secured: {e}")

    async def fetch(self, url: str):
        """Ensure VPN is active before using BasicSpider fetch."""
        print(f"[StealthSpider] Enforcing VPN for region: {self.region}")
        await self._ensure_stealth()
        return await super().fetch(url)
