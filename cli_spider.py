import argparse
import asyncio
import json
from pathlib import Path

from spider_core.browser.playwright_client import PlaywrightBrowserClient
from spider_core.core_utils.chunking import TextChunker
from spider_core.llm.openai_gpt_client import OpenAIGPTClient
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.spiders.basic_spider import BasicSpider

# ✅ Import StealthSpider and config defaults
try:
    from spider_core.spiders.stealth.stealth_spider import StealthSpider
    from spider_core.spiders.stealth.stealth_config import (
        DEFAULT_REGION,
        DEFAULT_VPN_PROVIDER,
        REQUIRE_VPN_DEFAULT,
    )
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False


async def run_spider(spider, url, output_path, pretty):
    print(f"🔍 Fetching: {url} using {spider.__class__.__name__} ...")
    result = await spider.fetch(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        json.dump(
            result.__dict__,
            f,
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
            indent=4 if pretty else None,
        )
        f.write("\n")
    print("\n--- Summary ---")
    print(spider.summarize_result(result))
    print("----------------")

    print(f"✅ Saved result to {output_path}")
    

def main():
    parser = argparse.ArgumentParser(description="LLM-powered web spider CLI.")

    parser.add_argument("url", help="URL to crawl")
    parser.add_argument("--output", default="output.jsonl", help="JSONL output file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max tokens per chunk")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in visible mode")

    # ✅ Stealth mode options
    parser.add_argument("--stealth", action="store_true", help="Use StealthSpider with VPN enforcement")
    parser.add_argument("--vpn", type=str, default=None, help="VPN provider (default: nordvpn)")
    parser.add_argument("--region", type=str, default=None, help="Region to connect VPN (e.g. hong_kong)")
    parser.add_argument("--no-require-vpn", action="store_true", help="Do not fail if VPN is not secured")

    args = parser.parse_args()

    async def async_main():
        # Init dependencies
        browser = PlaywrightBrowserClient(headless=not args.no_headless)
        llm = OpenAIGPTClient()
        ranker = RelevanceRanker(llm)
        chunker = TextChunker(max_tokens=args.max_tokens)

        # ✅ Select spider type
        if args.stealth:
            if not STEALTH_AVAILABLE:
                raise RuntimeError("StealthSpider is not available. Ensure stealth module is installed.")

            vpn_provider = args.vpn or DEFAULT_VPN_PROVIDER
            region = args.region or DEFAULT_REGION
            require_vpn = not args.no_require_vpn

            print(f"[CLI] Using StealthSpider with VPN={vpn_provider}, region={region}, require_vpn={require_vpn}")
            spider = StealthSpider(
                browser_client=browser,
                relevance_ranker=ranker,
                chunker=chunker,
                vpn_provider=vpn_provider,
                region=region,
                require_vpn=require_vpn,
            )
        else:
            print("[CLI] Using BasicSpider (no VPN enforcement).")
            spider = BasicSpider(browser, ranker, chunker)

        try:
            await run_spider(spider, args.url, Path(args.output), args.pretty)
        finally:
            # ✅ Ensures browser always closes on SAME event loop
            await browser.close()

    # ✅ Run within ONE consistent event loop
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
