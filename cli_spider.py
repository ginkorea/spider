import argparse
import asyncio
import json
from pathlib import Path

from spider_core.browser.playwright_client import PlaywrightBrowserClient
from spider_core.core_utils.chunking import TextChunker
from spider_core.llm.openai_gpt_client import OpenAIGPTClient
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.spiders.basic_spider import BasicSpider

# ✅ Optional stealth imports
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

# ✅ Goal-oriented modules
try:
    from spider_core.goal.goal_planner import GoalPlanner
    from spider_core.spiders.goal_spider import GoalOrientedSpider
    from spider_core.storage.db import DB
    GOAL_AVAILABLE = True
except ImportError:
    GOAL_AVAILABLE = False


# -----------------------------------------------------------------------------
# Core runner for single-page spiders
# -----------------------------------------------------------------------------
async def run_basic_spider(spider, url, output_path, pretty):
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


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM-powered web spider CLI")

    # Standard arguments
    parser.add_argument("url", help="Seed URL to crawl or fetch")
    parser.add_argument("--output", default="output.jsonl", help="JSONL output file path (basic mode)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max tokens per chunk")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in visible mode")

    # Stealth mode options
    parser.add_argument("--stealth", action="store_true", help="Use StealthSpider with VPN enforcement")
    parser.add_argument("--vpn", type=str, default=None, help="VPN provider (default: nordvpn)")
    parser.add_argument("--region", type=str, default=None, help="VPN region (e.g. hong_kong)")
    parser.add_argument("--no-require-vpn", action="store_true", help="Do not fail if VPN not connected")

    # Goal-driven mode options
    parser.add_argument("--goal", type=str, default=None, help="Goal/question to answer")
    parser.add_argument("--db", type=str, default="spider_core.db", help="SQLite database path for crawl data")
    parser.add_argument("--max-pages", type=int, default=20, help="Max pages to visit in goal mode")
    parser.add_argument("--confidence", type=float, default=0.85, help="Confidence threshold to stop in goal mode")

    args = parser.parse_args()

    async def async_main():
        browser = PlaywrightBrowserClient(headless=not args.no_headless)
        llm = OpenAIGPTClient()
        ranker = RelevanceRanker(llm)
        chunker = TextChunker(max_tokens=args.max_tokens)

        # -----------------------------------------------------------------------------
        # GOAL MODE
        # -----------------------------------------------------------------------------
        if args.goal:
            if not GOAL_AVAILABLE:
                raise RuntimeError("Goal modules not found. Ensure goal_spider.py, goal_planner.py, and storage/db.py exist.")

            print(f"[CLI] 🚀 Running Goal-Oriented Spider for goal: '{args.goal}'")
            db = DB(args.db)
            planner = GoalPlanner(llm)

            # Determine spider base
            if args.stealth:
                if not STEALTH_AVAILABLE:
                    raise RuntimeError("StealthSpider is not available. Install stealth module.")
                vpn_provider = args.vpn or DEFAULT_VPN_PROVIDER
                region = args.region or DEFAULT_REGION
                require_vpn = not args.no_require_vpn
                print(f"[CLI] Using StealthSpider VPN={vpn_provider}, region={region}, require_vpn={require_vpn}")
                base_spider = StealthSpider(
                    browser_client=browser,
                    relevance_ranker=ranker,
                    chunker=chunker,
                    vpn_provider=vpn_provider,
                    region=region,
                    require_vpn=require_vpn,
                )
            else:
                base_spider = BasicSpider(browser, ranker, chunker)

            goal_spider = GoalOrientedSpider(
                browser_client=browser,
                relevance_ranker=ranker,
                chunker=chunker,
                planner=planner,
                db=db,
                stop_threshold=args.confidence,
                max_pages=args.max_pages,
            )

            try:
                result = await goal_spider.fetch_goal(args.url, args.goal)
                print("\n=== GOAL RESULT ===")
                print(f"Goal: {result['goal']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Visited pages: {result['visited_count']}")
                print("\nAnswer:")
                print(result["answer"][:2000], "..." if len(result["answer"]) > 2000 else "", sep="")
                print("===================")
            finally:
                await browser.close()
                db.close()
            return

        # -----------------------------------------------------------------------------
        # BASIC OR STEALTH MODE
        # -----------------------------------------------------------------------------
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
            print("[CLI] Using BasicSpider (no VPN).")
            spider = BasicSpider(browser, ranker, chunker)

        try:
            await run_basic_spider(spider, args.url, Path(args.output), args.pretty)
        finally:
            await browser.close()

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
