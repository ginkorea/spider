# stealth/stealth_config.py

"""
Configuration mappings and defaults for StealthSpider VPN/Tor regions.
"""

# Default region if none specified
DEFAULT_REGION = "hong_kong"

# Supported region aliases -> NordVPN region codes
SUPPORTED_REGIONS = {
    "hong_kong": "hong_kong",
    "hk": "hong_kong",
    "japan": "japan",
    "jp": "japan",
    "korea": "south_korea",
    "kr": "south_korea",
    "taiwan": "taiwan",
    "tw": "taiwan",
    "singapore": "singapore",
    "sg": "singapore",
}

# Default VPN provider
DEFAULT_VPN_PROVIDER = "nordvpn"

# Default VPN requirement behavior
REQUIRE_VPN_DEFAULT = True

# IP intel check endpoint (used during verification)
IP_CHECK_URL = "https://ipinfo.io/json"
