# stealth/vpn_manager.py

import subprocess
import json
import urllib.request
from spider_core.spiders.stealth.stealth_config import SUPPORTED_REGIONS, IP_CHECK_URL


class NordVPNManager:
    """
    Handles connecting and verifying NordVPN sessions.
    """

    def connect(self, region: str):
        """
        Attempt to connect NordVPN to a specific region.
        Region must be a valid NordVPN location string.
        """
        try:
            subprocess.run(["nordvpn", "connect", region], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to connect NordVPN to region '{region}': {e}")

    def get_current_ipinfo(self) -> dict:
        """
        Return a JSON dict of IP info from external service.
        Can include region, country, IP, etc.
        """
        with urllib.request.urlopen(IP_CHECK_URL) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def is_in_region(self, region_key: str) -> bool:
        """
        Check if current public IP maps to the requested region by keyword match.
        """
        if region_key not in SUPPORTED_REGIONS:
            raise ValueError(f"Unsupported region key '{region_key}'")

        region_label = SUPPORTED_REGIONS[region_key]
        ipinfo = self.get_current_ipinfo()

        # We check if region/country string includes our region keyword (loose match).
        location = (ipinfo.get("region", "") + " " + ipinfo.get("country", "")).lower()
        return region_label.replace("_", " ") in location or region_label.split("_")[0] in location

    def ensure_region(self, region_key: str):
        """
        Ensure VPN is currently connected to the desired region.
        If not, attempt connection.
        """
        if not self.is_in_region(region_key):
            region_label = SUPPORTED_REGIONS[region_key]
            self.connect(region_label)
            if not self.is_in_region(region_key):
                raise RuntimeError(f"Failed to ensure NordVPN is connected to {region_key}.")
