import argparse
import asyncio
import json
from pathlib import Path
from browser.playwright_client import PlaywrightBrowserClient
from core_utils.chunking import TextChunker
from llm.openai_gpt_client import OpenAIGPTClient
from llm.relevance_ranker import RelevanceRanker
from spiders.basic_spider import BasicSpider


async def run_spider(url: str, output_path: Path, pretty: bool, max_tokens: int, headless: bool):
    browser = PlaywrightBrowserClient(headless=headless)
    llm = OpenAIGPTClient()
    ranker = RelevanceRanker(llm)
    chunker = TextChunker(max_tokens=max_tokens)
    spider = BasicSpider(browser, ranker, chunker)

    print(f"🔍 Fetching: {url} ...")
    result = await spider.fetch(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a") as f:
        json.dump(
            result.__dict__, 
            f, 
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
            indent=4 if pretty else None
        )
        f.write("\n")

    print(f"✅ Saved result to {output_path}")
    await browser.close()


def main():
    parser = argparse.ArgumentParser(description="LLM-powered web spider CLI.")
    parser.add_argument("--url", required=True, help="URL to crawl")
    parser.add_argument("--output", default="output.jsonl", help="JSONL output file path")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max tokens per chunk")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in visible mode")

    args = parser.parse_args()

    asyncio.run(run_spider(
        url=args.url,
        output_path=Path(args.output),
        pretty=args.pretty,
        max_tokens=args.max_tokens,
        headless=not args.no_headless
    ))


if __name__ == "__main__":
    main()
