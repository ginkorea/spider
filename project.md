# Project Compilation: spider_core

## 🧾 Summary

| Metric | Value |
|:--|:--|
| Root Directory | `/home/gompert/data/workspace/spider_core` |
| Total Directories | 10 |
| Total Indexed Files | 39 |
| Skipped Files | 0 |
| Indexed Size | 79.99 KB |
| Max File Size Limit | 2 MB |

## 📚 Table of Contents

- [.__init__.py.swp](#init-py-swp)
- [README.md](#readme-md)
- [__init__.py](#init-py)
- [base/__init__.py](#base-init-py)
- [base/link_metadata.py](#base-link-metadata-py)
- [base/page_result.py](#base-page-result-py)
- [base/spider.py](#base-spider-py)
- [browser/__init__.py](#browser-init-py)
- [browser/browser_client.py](#browser-browser-client-py)
- [browser/playwright_client.py](#browser-playwright-client-py)
- [cli_spider.py](#cli-spider-py)
- [core_utils/__init__.py](#core-utils-init-py)
- [core_utils/chunking.py](#core-utils-chunking-py)
- [core_utils/url_utils.py](#core-utils-url-utils-py)
- [extractors/__init__.py](#extractors-init-py)
- [extractors/deterministic_extractor.py](#extractors-deterministic-extractor-py)
- [goal/__init__.py](#goal-init-py)
- [goal/goal_planner.py](#goal-goal-planner-py)
- [llm/__init__.py](#llm-init-py)
- [llm/embeddings_client.py](#llm-embeddings-client-py)
- [llm/llm_client.py](#llm-llm-client-py)
- [llm/openai_gpt_client.py](#llm-openai-gpt-client-py)
- [llm/relevance_ranker.py](#llm-relevance-ranker-py)
- [requirements.txt](#requirements-txt)
- [spiders/__init__.py](#spiders-init-py)
- [spiders/basic_spider.py](#spiders-basic-spider-py)
- [spiders/goal_spider.py](#spiders-goal-spider-py)
- [spiders/stealth/__init__.py](#spiders-stealth-init-py)
- [spiders/stealth/stealth_config.py](#spiders-stealth-stealth-config-py)
- [spiders/stealth/vpn_manager.py](#spiders-stealth-vpn-manager-py)
- [spiders/stealth_spider.py](#spiders-stealth-spider-py)
- [storage/__init__.py](#storage-init-py)
- [storage/db.py](#storage-db-py)
- [test/test.py](#test-test-py)
- [test/test2.py](#test-test2-py)
- [test/test3.py](#test-test3-py)
- [test/test4.py](#test-test4-py)
- [test/test_ranker.py](#test-test-ranker-py)
- [test/test_spider.py](#test-test-spider-py)

## 📂 Project Structure

```
📁 base/
    📄 __init__.py
    📄 link_metadata.py
    📄 page_result.py
    📄 spider.py
📁 browser/
    📄 __init__.py
    📄 browser_client.py
    📄 playwright_client.py
📁 core_utils/
    📄 __init__.py
    📄 chunking.py
    📄 url_utils.py
📁 extractors/
    📄 __init__.py
    📄 deterministic_extractor.py
📁 goal/
    📄 __init__.py
    📄 goal_planner.py
📁 llm/
    📄 __init__.py
    📄 embeddings_client.py
    📄 llm_client.py
    📄 openai_gpt_client.py
    📄 relevance_ranker.py
📁 spiders/
    📁 stealth/
        📄 __init__.py
        📄 stealth_config.py
        📄 vpn_manager.py
    📄 __init__.py
    📄 basic_spider.py
    📄 goal_spider.py
    📄 stealth_spider.py
📁 storage/
    📄 __init__.py
    📄 db.py
📁 test/
    📄 test.py
    📄 test2.py
    📄 test3.py
    📄 test4.py
    📄 test_ranker.py
    📄 test_spider.py
📄 __init__.py
📄 cli_spider.py
📄 README.md
📄 requirements.txt
```

## `README.md`

```markdown
# 🕷️ go-spider

**go-spider** is an LLM-driven, goal-oriented web crawler designed for autonomous information discovery.  
It combines browser-based rendering, intelligent link prioritization, and language-model reasoning to search the web until it achieves a defined goal — not just until it runs out of links.

---

## 🚀 Key Features

- **🎯 Goal-Oriented Crawling** – Specify a goal or question (e.g. “Find this company’s investor relations contact”) and let go-spider autonomously explore until it finds a confident answer.
- **🧠 LLM-Powered Relevance Ranking** – Uses a local or cloud-based LLM to score link and content relevance dynamically.
- **🕵️ Stealth Mode (Optional)** – Integrates with VPNs for geo-specific or anonymized browsing.
- **📚 Persistent Memory** – Optionally logs pages and embeddings for later retrieval-augmented generation (RAG) search.
- **🌐 Playwright Rendering** – Fetches fully rendered pages (JS, AJAX, etc.) to ensure accurate content capture.
- **⚙️ Modular Architecture** – Swap in your own LLM client, browser backend, or vector DB.

---

## 🧩 Installation

Install directly from PyPI:

```bash
pip install go-spider
````

To run locally with development dependencies:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/go-spider.git
cd go-spider
pip install -e .
```

---

## 🕹️ Basic Usage

### Crawl a single site

```bash
spider "https://example.com"
```

### Goal-Oriented Crawl

```bash
spider "https://example.com" \
  --goal "Find this site's investor relations email" \
  --max-pages 25 \
  --confidence 0.9
```

Example output:

```
[CLI] 🚀 Running Goal Spider for goal: 'Find this site's investor relations email'
[BasicSpider] INFO Rendering URL: https://example.com
[BasicSpider] INFO Rendering URL: https://iana.org/domains/example
[BasicSpider] INFO Rendering URL: https://iana.org/contact

=== GOAL RESULT ===
Goal: Find this site's investor relations email
Confidence: 1.00
Visited pages: 3
Answer:
No contact email found.
The page does not provide a direct contact email, but includes: iana@iana.org
===================
```

---

## ⚙️ Command-Line Options

| Flag            | Description                                    |
| --------------- | ---------------------------------------------- |
| `--goal`        | Question or target objective for the crawl     |
| `--stealth`     | Use VPN-protected stealth browsing             |
| `--region`      | VPN region (e.g. `hong_kong`)                  |
| `--db`          | SQLite database path for persistent crawl logs |
| `--max-pages`   | Maximum number of pages to visit               |
| `--confidence`  | Confidence threshold to stop searching         |
| `--pretty`      | Pretty-print output JSON                       |
| `--no-headless` | Run Playwright browser visibly                 |
| `--output`      | Output path (default: `output.jsonl`)          |

---

## 🧠 Architecture Overview

```text
PlaywrightBrowserClient  →  BasicSpider / StealthSpider / GoalSpider
                                    ↓
                         RelevanceRanker (LLM-based)
                                    ↓
                               TextChunker
                                    ↓
                              Goal Planner
                                    ↓
                             SQLite + Embeddings
```

* **BasicSpider** – standard site fetcher
* **StealthSpider** – uses VPN-enforced browsing
* **GoalSpider** – iterative goal-driven crawler (core of go-spider)

---

## 🧩 Integration Example (Python API)

You can also use it as a library:

```python
from spider_core.spiders.goal_spider import GoalSpider
from spider_core.browser.playwright_client import PlaywrightBrowserClient
from spider_core.llm.openai_gpt_client import OpenAIGPTClient
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.core_utils.chunking import TextChunker
import asyncio

async def main():
    browser = PlaywrightBrowserClient()
    llm = OpenAIGPTClient()
    ranker = RelevanceRanker(llm)
    chunker = TextChunker()

    spider = GoalSpider(browser_client=browser, relevance_ranker=ranker, chunker=chunker, llm_client=llm)
    result = await spider.run_goal("https://example.com", "Find contact email")
    print(result)

asyncio.run(main())
```

---

## 🧰 Requirements

* Python 3.10+
* Playwright
* OpenAI-compatible LLM API key (optional)
* Works on Linux, macOS, and Windows (with Playwright browsers installed)

---

## 🏗️ Roadmap

* [ ] Add distributed crawling support
* [ ] Integrate local LLMs via `llama.cpp`
* [ ] Add vector DB backends (FAISS / Chroma / SQLite-VSS)
* [ ] Fine-grained crawl policies (robots.txt, depth weighting)
* [ ] RAG API for querying past crawls

---

## 🧑‍💻 Author

**Josh Gompert**
AI Systems Engineer • Data Scientist • Information Operations Officer

* GitHub: [@ginkorea](https://github.com/ginkorea)
* PyPI: [go-spider](https://pypi.org/project/go-spider/)

---

## 🪪 License

**MIT License** — free to use, modify, and distribute.
See `LICENSE` file for details.

---

> *“go-spider doesn’t just crawl the web — it pursues intent.”*

```

## `__init__.py`

```python
"""
spider_core
-----------
Main package for the goal-oriented LLM web spider system.
"""
__all__ = ["spiders", "browser", "goal", "llm", "storage", "base", "core_utils", "extractors"]

```

## `base/__init__.py`

```python
"""
spider_core.base
----------------
Base interfaces and data models used by all spiders.
"""

from spider_core.base.spider import Spider
from spider_core.base.page_result import PageResult
from spider_core.base.link_metadata import LinkMetadata

__all__ = ["Spider", "PageResult", "LinkMetadata"]

```

## `base/link_metadata.py`

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LinkMetadata:
    href: str
    text: Optional[str]
    rel: List[str]
    detected_from: List[str]
    llm_score: float = 0.0
    llm_tags: Optional[List[str]] = None
    reasons: Optional[List[str]] = None

```

## `base/page_result.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from spider_core.base.link_metadata import LinkMetadata


@dataclass
class PageResult:
    url: str
    fetched_at: datetime
    status: int
    canonical: str
    links: List[LinkMetadata]
    llm_summary: Optional[str]
    page_chunks: List[Dict]

```

## `base/spider.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Spider(ABC):
    """
    Abstract base spider contract.
    All spiders must implement `fetch()` that returns a structured PageResult.
    """

    @abstractmethod
    async def fetch(self, url: str) -> Dict[str, Any]:
        """Fetch a URL and return structured data (PageResult)."""
        raise NotImplementedError("All spiders must implement fetch()")

```

## `browser/__init__.py`

```python

```

## `browser/browser_client.py`

```python
from abc import ABC, abstractmethod
from typing import Tuple


class BrowserClient(ABC):
    """
    Abstract base for a browser client.
    """

    @abstractmethod
    async def render(self, url: str) -> Tuple[str, str, int]:
        """
        Render a page and return:
          - html (str)
          - visible_text (str)
          - status_code (int)
        """
        pass

```

## `browser/playwright_client.py`

```python
import asyncio
from typing import Tuple
from playwright.async_api import async_playwright, Browser, Page
from spider_core.browser.browser_client import BrowserClient


class PlaywrightBrowserClient(BrowserClient):
    """
    A Playwright-based implementation of BrowserClient.
    Launches Chromium and renders pages asynchronously.
    """

    def __init__(self, headless: bool = True, viewport: Tuple[int, int] = (1200, 900)):
        self.headless = headless
        self.viewport = viewport
        self.browser: Browser | None = None

    async def _ensure_browser(self):
        """Launches the browser if it's not already running."""
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=self.headless)

    async def render(self, url: str) -> Tuple[str, str, int]:
        """
        Render a page and return:
          - html (str)
          - visible_text (str)
          - status_code (int)
        """
        await self._ensure_browser()
        page: Page = await self.browser.new_page(viewport={"width": self.viewport[0], "height": self.viewport[1]})

        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        status_code = response.status if response else 0

        # Optional scroll to trigger lazy-loaders
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2);")
            await asyncio.sleep(0.5)
        except Exception:
            pass

        html = await page.content()

        # Visible text (scripts/styles removed by Playwright's innerText handling)
        visible_text = await page.evaluate("""
            () => {
                const clone = document.body.cloneNode(true);
                clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());
                return clone.innerText;
            }
        """)

        await page.close()
        return html, visible_text, status_code

    async def close(self):
        """Close the browser when done."""
        if self.browser is not None:
            await self.browser.close()
            self.browser = None

```

## `cli_spider.py`

```python
import argparse
import asyncio
import json
from pathlib import Path

from spider_core.browser.playwright_client import PlaywrightBrowserClient
from spider_core.core_utils.chunking import TextChunker
from spider_core.llm.openai_gpt_client import OpenAIGPTClient
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.spiders.basic_spider import BasicSpider

# ---------------------------------------------------------------------
# Optional stealth mode imports
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Optional goal-oriented imports
# ---------------------------------------------------------------------
try:
    # NOTE: your module currently defines GoalSpider, not GoalOrientedSpider
    from spider_core.spiders.goal_spider import GoalSpider
    GOAL_AVAILABLE = True
except ImportError:
    GOAL_AVAILABLE = False


# ---------------------------------------------------------------------
# Helper: run a simple single-page spider (basic/stealth)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM-powered recursive web spider CLI")

    # Base arguments
    parser.add_argument("url", help="Seed URL to crawl or fetch")
    parser.add_argument("--output", default="output.jsonl", help="JSONL output path for basic mode")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")
    parser.add_argument("--max-tokens", type=int, default=1200, help="Max tokens per chunk")
    parser.add_argument("--no-headless", action="store_true", help="Run Playwright in visible (non-headless) mode")

    # Stealth mode arguments
    parser.add_argument("--stealth", action="store_true", help="Use StealthSpider with VPN enforcement")
    parser.add_argument("--vpn", type=str, default=None, help="VPN provider (default: nordvpn)")
    parser.add_argument("--region", type=str, default=None, help="VPN region (e.g. hong_kong)")
    parser.add_argument("--no-require-vpn", action="store_true", help="Do not fail if VPN not connected")

    # Goal-oriented mode arguments
    parser.add_argument("--goal", type=str, default=None, help="Goal or question to recursively answer")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum recursion depth for goal mode")
    # keep these args for future; currently not used by GoalSpider, but harmless:
    parser.add_argument("--db", type=str, default="spider_core.db", help="(Reserved) SQLite DB path for future use")
    parser.add_argument("--max-pages", type=int, default=25, help="(Reserved) Max pages to crawl in goal mode")
    parser.add_argument("--confidence", type=float, default=0.85, help="(Reserved) Confidence threshold for goal mode")

    args = parser.parse_args()

    async def async_main():
        # -----------------------------------------------------------------
        # Shared components
        # -----------------------------------------------------------------
        browser = PlaywrightBrowserClient(headless=not args.no_headless)
        llm = OpenAIGPTClient()
        ranker = RelevanceRanker(llm)
        chunker = TextChunker(max_tokens=args.max_tokens)

        # -----------------------------------------------------------------
        # GOAL-ORIENTED MODE
        # -----------------------------------------------------------------
        if args.goal:
            if not GOAL_AVAILABLE:
                raise RuntimeError("Goal modules not found. Ensure spiders/goal_spider.py exists and is importable.")

            print(f"[CLI] 🚀 Running Goal Spider for goal: '{args.goal}'")

            # Choose base spider (stealth or normal) just for behavior parity;
            # GoalSpider itself will handle recursive crawling.
            if args.stealth:
                if not STEALTH_AVAILABLE:
                    raise RuntimeError("StealthSpider not available. Install stealth module.")
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
                # We don't actually *use* base_spider inside GoalSpider yet, but you can wire it in later.
            else:
                base_spider = BasicSpider(browser, ranker, chunker)

            # Initialize the goal spider (uses llm directly)
            goal_spider = GoalSpider(
                browser_client=browser,
                relevance_ranker=ranker,
                chunker=chunker,
                llm_client=llm,
                max_depth=args.max_depth,
            )

            try:
                result = await goal_spider.run_goal(args.url, args.goal)
                print("\n=== GOAL RESULT ===")
                print(f"Goal: {result['goal']}")
                print(f"Found: {result['found']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Visited pages: {result['visited_pages']}")
                print("\nAnswer:")
                print(result["answer"] or "(no answer found)")
                print("===================")
            finally:
                await browser.close()
            return

        # -----------------------------------------------------------------
        # BASIC OR STEALTH MODE
        # -----------------------------------------------------------------
        if args.stealth:
            if not STEALTH_AVAILABLE:
                raise RuntimeError("StealthSpider not available. Ensure stealth module installed.")
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

```

## `core_utils/__init__.py`

```python
"""
spider_core.core_utils
----------------------
Utility functions for URL normalization, chunking, and tokenization.
"""

from spider_core.core_utils.chunking import TextChunker
from spider_core.core_utils.url_utils import canonicalize_url

__all__ = ["TextChunker", "canonicalize_url"]

```

## `core_utils/chunking.py`

```python
import tiktoken
from typing import List, Dict


class TextChunker:
    """
    Splits large text into LLM-friendly chunks with estimated token limits.
    Attempts to preserve paragraph structure for coherence.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1200):
        """
        :param model: GPT model name for tokenizer.
        :param max_tokens: Max tokens per chunk.
        """
        self.max_tokens = max_tokens

        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        return len(self.encoder.encode(text))

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Returns a list of chunk objects: {"chunk_id", "text", "token_count"}
        Large paragraphs will be split if necessary.
        """
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current_text = ""
        current_tokens = 0
        chunk_id = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If this paragraph alone exceeds max tokens, split by words
            if para_tokens > self.max_tokens:
                words = para.split()
                sub_text = ""
                for word in words:
                    test_text = (sub_text + " " + word).strip()
                    if self.count_tokens(test_text) > self.max_tokens:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": sub_text.strip(),
                            "token_count": self.count_tokens(sub_text)
                        })
                        chunk_id += 1
                        sub_text = word
                    else:
                        sub_text = test_text

                if sub_text.strip():
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": sub_text.strip(),
                        "token_count": self.count_tokens(sub_text)
                    })
                    chunk_id += 1

                continue

            # Try adding the paragraph to current chunk
            if current_tokens + para_tokens <= self.max_tokens:
                current_text += ("\n\n" + para if current_text else para)
                current_tokens += para_tokens
            else:
                # Finalize current chunk & start a new one
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_text,
                    "token_count": current_tokens
                })
                chunk_id += 1
                current_text = para
                current_tokens = para_tokens

        # Add the last chunk if any
        if current_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_text,
                "token_count": current_tokens
            })

        return chunks

```

## `core_utils/url_utils.py`

```python
from urllib.parse import urljoin, urlparse, urlunparse


def canonicalize_url(href: str, base_url: str) -> str | None:
    """
    Convert a possibly relative URL into an absolute canonical form.
    Removes URL fragments. Returns None if invalid.
    """
    try:
        # Join with base (relative → absolute)
        absolute = urljoin(base_url, href)

        # Parse and remove fragment (#...)
        parsed = urlparse(absolute)
        cleaned = parsed._replace(fragment="")

        return urlunparse(cleaned)
    except Exception:
        return None

```

## `extractors/__init__.py`

```python
"""
spider_core.extractors
----------------------
HTML content and link extraction modules.
"""

from spider_core.extractors.deterministic_extractor import DeterministicLinkExtractor

__all__ = ["DeterministicLinkExtractor"]

```

## `extractors/deterministic_extractor.py`

```python
from typing import List
from bs4 import BeautifulSoup
from spider_core.base.link_metadata import LinkMetadata
from spider_core.core_utils.url_utils import canonicalize_url


class DeterministicLinkExtractor:
    """
    Deterministically extracts visible and metadata-based link structures.
    Does not rely on LLMs or heuristics – purely structural extraction.
    """

    @staticmethod
    def extract(html: str, base_url: str) -> List[LinkMetadata]:
        soup = BeautifulSoup(html, "lxml")
        link_map = {}  # href -> LinkMetadata (deduplicates canonical URLs)

        def add_link(raw_href: str, text: str | None, rel: list[str], source: str):
            canon = canonicalize_url(raw_href, base_url)
            if not canon:
                return

            # Create new or merge into existing record
            if canon not in link_map:
                link_map[canon] = LinkMetadata(
                    href=canon,
                    text=(text.strip()[:300] if text else None),
                    rel=rel or [],
                    detected_from=[source],
                    llm_score=0.0,
                    llm_tags=None,
                    reasons=None
                )
            else:
                entry = link_map[canon]
                # Add new source if missing
                if source not in entry.detected_from:
                    entry.detected_from.append(source)
                # Merge rel attributes
                for r in rel:
                    if r not in entry.rel:
                        entry.rel.append(r)
                # If no text set yet and this one has text, use it
                if not entry.text and text:
                    entry.text = text.strip()[:300]

        # 1️⃣ Extract <a href=""> links
        for a in soup.find_all("a", href=True):
            add_link(a["href"], a.get_text(), a.get("rel", []), "a")

        # 2️⃣ Extract <link> tags (e.g., canonical, preload, etc.)
        for link in soup.find_all("link", href=True):
            rel = link.get("rel", [])
            add_link(link["href"], None, rel, f"link:{','.join(rel) or 'link'}")

        # 3️⃣ Extract OpenGraph / Twitter metadata URLs
        for meta in soup.find_all("meta", content=True):
            prop = meta.get("property", "").lower()
            name = meta.get("name", "").lower()
            if prop == "og:url" or name in ("og:url", "twitter:url"):
                add_link(meta["content"], None, [], "meta")

        # 4️⃣ Extract data-href style links commonly used in JS menus
        for el in soup.find_all(attrs={"data-href": True}):
            add_link(el["data-href"], el.get_text(), [], "data-href")

        return list(link_map.values())

```

## `goal/__init__.py`

```python
"""
spider_core.goal
----------------
Goal planner for reasoning, retrieval, and LLM guidance.
"""

from spider_core.goal.goal_planner import GoalPlanner

__all__ = ["GoalPlanner"]

```

## `goal/goal_planner.py`

```python
# goal/goal_planner.py
from typing import List, Dict, Any, Tuple
from spider_core.llm.openai_gpt_client import OpenAIGPTClient

# ---------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------

GOAL_CHUNK_SYSTEM = (
    "You are a goal-driven web research planner. "
    "Given a user GOAL and a PAGE CHUNK, you will: "
    "1) estimate if the GOAL is fully answered by this page content (0-1), "
    "2) extract a concise answer delta (new facts that progress the goal), "
    "3) propose next links (subset of candidates) most likely to progress the goal.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "{"
    '  "goal_satisfaction_estimate": float (0..1),'
    '  "answer_delta": "short string",'
    '  "next_link_scores": [{"href": "...", "score": float (0..1)}]'
    "}"
)

GOAL_CONTEXT_SYSTEM = (
    "You are an evaluator for a goal-oriented web crawler.\n"
    "Your job is to decide whether the user's GOAL has been answered by the given CONTEXT.\n\n"
    "Rules:\n"
    "- Be strict but pragmatic: if the context clearly contains the answer, mark found=true.\n"
    "- 'confidence' is a float between 0 and 1 expressing how sure you are that the GOAL is satisfied.\n"
    "- 'answer' should be a short, direct answer in natural language, using only information from CONTEXT.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "{"
    '  "found": bool,'
    '  "confidence": float (0..1),'
    '  "answer": "short string"'
    "}"
)

LINK_PLANNER_SYSTEM = (
    "You are a navigation planner for a generic goal-oriented web crawler.\n"
    "Given a GOAL, the CURRENT PAGE URL, a PAGE SNIPPET, the current CONFIDENCE, and a list of CANDIDATE LINKS, "
    "you will assign each link a score from 0.0 to 1.0 indicating how promising it is for making progress "
    "toward the GOAL.\n\n"
    "You must handle arbitrary goals (not just contact info): product questions, policies, biographies, etc.\n"
    "Prefer links that:\n"
    "- are semantically related to the GOAL based on their anchor text and href,\n"
    "- look like high-level information or overview pages when the GOAL is broad,\n"
    "- look like specific detail pages (FAQ, docs, terms, policies, etc.) when the GOAL is narrow.\n\n"
    "Return ONLY valid JSON of the form:\n"
    "{"
    '  "link_scores": ['
    '    {"href": "https://example.com/path", "score": 0.0},'
    '    ...'
    "  ]"
    "}"
)


def build_chunk_prompt(goal: str, chunk_text: str, link_candidates: List[Dict[str, str]]) -> str:
    return (
        f"GOAL:\n{goal}\n\n"
        f"PAGE CHUNK (truncated):\n{chunk_text[:4000]}\n\n"
        f"LINK CANDIDATES (href + text):\n"
        f"{[{'href': l['href'], 'text': l.get('text', '')[:200]} for l in link_candidates]}\n\n"
        "Return JSON as described."
    )


class GoalPlanner:
    """
    High-level planner for goal-oriented crawling.

    Responsibilities:
      - Evaluate whether a given CONTEXT (RAG-retrieved text) satisfies the GOAL.
      - Score links on a page according to how promising they are for the GOAL.
      - (Legacy) evaluate individual chunks for goal satisfaction and per-link scores.
    """

    def __init__(self, llm: OpenAIGPTClient):
        self.llm = llm

    # ------------------------------------------------------------------
    # Legacy / chunk-based interface (kept for compatibility)
    # ------------------------------------------------------------------
    async def evaluate_chunk(
        self,
        goal: str,
        chunk_text: str,
        link_candidates: List[Dict[str, Any]],
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Evaluate a single PAGE CHUNK relative to GOAL.

        Returns:
          - goal_satisfaction_estimate (0..1)
          - answer_delta (short string)
          - next_link_scores: {href -> score}
        """
        prompt = build_chunk_prompt(goal, chunk_text, link_candidates)
        out = await self.llm.complete_json(GOAL_CHUNK_SYSTEM, prompt)

        est = float(out.get("goal_satisfaction_estimate", 0.0))
        delta = out.get("answer_delta", "").strip()
        next_scores_raw = out.get("next_link_scores", []) or []
        scored = {}
        for x in next_scores_raw:
            href = x.get("href")
            if not href:
                continue
            try:
                s = float(x.get("score", 0.0))
            except (TypeError, ValueError):
                s = 0.0
            scored[href] = s

        return est, delta, scored

    # ------------------------------------------------------------------
    # NEW: Context-level goal evaluation (RAG)
    # ------------------------------------------------------------------
    async def evaluate_context(self, goal: str, context_text: str) -> Tuple[bool, float, str]:
        """
        Given a GOAL and aggregated CONTEXT (e.g. top-k retrieved chunks),
        decide whether the GOAL is satisfied.

        Returns:
          - found: bool
          - confidence: float (0..1)
          - answer: short string
        """
        # Don't waste tokens if there's no context at all
        if not context_text.strip():
            return False, 0.0, ""

        prompt = (
            f"GOAL:\n{goal}\n\n"
            f"CONTEXT (truncated if long):\n{context_text[:8000]}"
        )

        out = await self.llm.complete_json(GOAL_CONTEXT_SYSTEM, prompt)
        found = bool(out.get("found", False))
        try:
            confidence = float(out.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        answer = out.get("answer", "").strip()

        return found, confidence, answer

    # ------------------------------------------------------------------
    # NEW: Generic, goal-aware link scoring
    # ------------------------------------------------------------------
    async def score_links(
        self,
        goal: str,
        page_url: str,
        page_snippet: str,
        link_candidates: List[Dict[str, str]],
        current_confidence: float,
        max_links: int = 50,
    ) -> Dict[str, float]:
        """
        Score candidate links for their usefulness toward achieving GOAL.

        link_candidates: list of dicts with keys "href", "text".
        Returns: {href -> score (0..1)}.
        """
        if not link_candidates:
            return {}

        # truncate to keep token usage sane
        trimmed = link_candidates[:max_links]
        page_snippet = (page_snippet or "")[:1500]

        user_prompt = (
            f"GOAL:\n{goal}\n\n"
            f"CURRENT CONFIDENCE:\n{current_confidence}\n\n"
            f"CURRENT PAGE URL:\n{page_url}\n\n"
            f"PAGE SNIPPET:\n{page_snippet}\n\n"
            f"CANDIDATE LINKS (href + text):\n"
            f"{[{'href': l.get('href'), 'text': (l.get('text') or '')[:200]} for l in trimmed]}\n\n"
            "Assign each link a 'score' from 0.0 to 1.0 indicating how promising it is for making progress "
            "toward the GOAL. You may give 0.0 to clearly irrelevant links."
        )

        try:
            out = await self.llm.complete_json(LINK_PLANNER_SYSTEM, user_prompt)
        except Exception:
            # Fail safe: no goal-aware scores
            return {}

        scores: Dict[str, float] = {}
        for item in out.get("link_scores", []) or []:
            href = item.get("href")
            if not href:
                continue
            try:
                s = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                s = 0.0
            scores[href] = s

        return scores

```

## `llm/__init__.py`

```python
"""
spider_core.llm
---------------
LLM-related clients and utilities (OpenAI, embeddings, ranking).
"""

from spider_core.llm.openai_gpt_client import OpenAIGPTClient
from spider_core.llm.relevance_ranker import RelevanceRanker
from spider_core.llm.embeddings_client import OpenAIEmbeddings, EmbeddingsClient

__all__ = [
    "OpenAIGPTClient",
    "RelevanceRanker",
    "OpenAIEmbeddings",
    "EmbeddingsClient",
]

```

## `llm/embeddings_client.py`

```python
# llm/embeddings_client.py
from abc import ABC, abstractmethod
from typing import List
import os
import numpy as np
import openai

class EmbeddingsClient(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

class OpenAIEmbeddings(EmbeddingsClient):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for embeddings.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(a @ b / (na * nb))

```

## `llm/llm_client.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClient(ABC):
    """
    Abstract base class for LLM interactions.
    """

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Send a request that expects a structured JSON response.
        """
        pass

```

## `llm/openai_gpt_client.py`

```python
import json
import asyncio
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
import openai


# ✅ Load environment variables
load_dotenv()  # Load project-level .env (if present)
load_dotenv(Path("~/.elf_env").expanduser(), override=False)  # Load personal fallback


class OpenAIGPTClient:
    """
    LLM client for GPT models using OpenAI's API (v2.x).
    Supports JSON-mode completions with retry.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # GPT-4.1-mini alias
        max_retries: int = 2,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ):
        # ✅ Prefer explicit API key > environment variables
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY in .env or ~/.elf_env")

        # ✅ v2.x uses a client object
        self.client = openai.OpenAI(api_key=api_key)

        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

    async def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Sends a JSON-enforced completion request.
        Tries up to `max_retries` times to parse valid JSON.
        Runs the sync OpenAI call in a background thread.
        """
        attempt = 0
        error_message = ""

        while attempt <= self.max_retries:
            try:
                # ✅ Run synchronous OpenAI call in async-safe thread
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                content = response.choices[0].message.content.strip()
                return json.loads(content)

            except Exception as e:
                attempt += 1
                error_message = str(e)
                await asyncio.sleep(0.5)

        raise RuntimeError(f"Failed to parse valid JSON after retries. Last error: {error_message}")

```

## `llm/relevance_ranker.py`

```python
import asyncio
from typing import List, Dict
from spider_core.base.link_metadata import LinkMetadata
from spider_core.llm.openai_gpt_client import OpenAIGPTClient


class RelevanceRanker:
    """
    Uses an LLM to evaluate the relevance of each link based on page chunks.
    """

    def __init__(self, llm_client: OpenAIGPTClient, max_reason_count: int = 3):
        self.llm_client = llm_client
        self.max_reason_count = max_reason_count

    async def score_links(self, links: List[LinkMetadata], chunks: List[Dict]) -> List[LinkMetadata]:
        """
        Scores links using chunk-based evaluation with GPT.
        """

        # Initialize aggregation structure
        scores = {link.href: {"sum": 0.0, "count": 0, "tags": set(), "reasons": []} for link in links}

        for chunk in chunks:
            system_prompt = (
                "You are an AI that evaluates the relevance of web links based on page content. "
                "Given a page text chunk and a set of candidate links (with href and anchor text), "
                "score each link from 0.0 to 1.0 based on how likely it is to be important or useful. "
                "Output JSON ONLY in this form: "
                '{"results":[{"href":"...", "score":0.0, "tags":["..."], "reason":"..."}]}'
            )

            candidate_minimal = [{"href": l.href, "text": l.text or ""} for l in links]

            user_prompt = (
                f"PAGE CHUNK:\n{chunk['text']}\n\n"
                f"LINK CANDIDATES:\n{candidate_minimal}\n\n"
                "Respond with relevance scores."
            )

            result = await self.llm_client.complete_json(system_prompt, user_prompt)

            if "results" in result and isinstance(result["results"], list):
                for item in result["results"]:
                    href = item.get("href")
                    if href in scores:
                        score = float(item.get("score", 0))
                        tags = item.get("tags", [])
                        reason = item.get("reason", "")

                        scores[href]["sum"] += score
                        scores[href]["count"] += 1
                        scores[href]["tags"].update(tags)
                        if reason:
                            scores[href]["reasons"].append(reason)

        # Apply aggregated scores to the LinkMetadata objects
        for link in links:
            data = scores[link.href]
            if data["count"] > 0:
                link.llm_score = round(data["sum"] / data["count"], 3)
                link.llm_tags = list(data["tags"])
                link.reasons = data["reasons"][: self.max_reason_count]

        return links

```

## `requirements.txt`

```text
playwright>=1.42.0
openai>=1.3.0
pydantic>=2.5.0
beautifulsoup4>=4.12.0
lxml>=4.9.3
tiktoken>=0.6.0

```

## `spiders/__init__.py`

```python
"""
spider_core.spiders
-------------------
All spider implementations (Basic, Stealth, Goal).
"""

from spider_core.spiders.basic_spider import BasicSpider
from spider_core.spiders.stealth_spider import StealthSpider
from spider_core.spiders.goal_spider import GoalSpider

__all__ = ["BasicSpider", "StealthSpider", "GoalSpider"]

```

## `spiders/basic_spider.py`

```python
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

```

## `spiders/goal_spider.py`

```python
import heapq
import logging
from typing import Dict, List, Optional, Set, Tuple
from spider_core.spiders.basic_spider import BasicSpider
from spider_core.base.page_result import PageResult
from spider_core.storage.db import DB
from spider_core.goal.goal_planner import GoalPlanner
from spider_core.llm.embeddings_client import EmbeddingsClient, OpenAIEmbeddings

logger = logging.getLogger("GoalSpider")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[GoalSpider] %(message)s"))
    logger.addHandler(ch)


class GoalSpider(BasicSpider):
    """RAG-augmented spider that reasons toward an arbitrary goal."""

    def __init__(
        self,
        browser_client,
        relevance_ranker,
        chunker,
        llm_client,
        db: Optional[DB] = None,
        planner: Optional[GoalPlanner] = None,
        embedder: Optional[EmbeddingsClient] = None,
        embed_model: str = "text-embedding-3-small",
        max_depth: int = 3,
        stop_threshold: float = 0.9,
        max_pages: int = 25,
        retrieval_top_k: int = 8,
    ):
        super().__init__(browser_client, relevance_ranker, chunker)
        self.llm = llm_client
        self.db = db or DB()
        self.planner = planner or GoalPlanner(self.llm)
        self.embed_model = embed_model
        self.embedder = embedder or OpenAIEmbeddings(model=embed_model)
        self.max_depth = max_depth
        self.stop_threshold = stop_threshold
        self.max_pages = max_pages
        self.retrieval_top_k = retrieval_top_k
        self.visited: Set[str] = set()

    async def fetch(self, url: str) -> PageResult:
        """Wrap BasicSpider.fetch() and add logging for goal context."""
        logger.info(f"[GoalSpider] Delegating fetch for: {url}")
        return await super().fetch(url)

    async def run_goal(self, start_url: str, goal: str) -> Dict:
        """Iterative crawl loop using RAG context evaluation."""
        heap: List[Tuple[float, str, int]] = [(-1.0, start_url, 0)]
        best_answer, best_conf = "", 0.0

        while heap and len(self.visited) < self.max_pages:
            found, conf, ans = await self._evaluate_memory(goal)
            if conf > best_conf:
                best_conf, best_answer = conf, ans
            if found and best_conf >= self.stop_threshold:
                logger.info(f"✅ Goal reached with confidence={best_conf:.2f}")
                break

            _, url, depth = heapq.heappop(heap)
            if url in self.visited or depth > self.max_depth:
                continue
            self.visited.add(url)

            logger.info(f"[Depth {depth}] Crawling {url}")
            page = await super().fetch(url)
            await self._store_page(page)

            scores = await self._score_links_for_goal(goal, page, best_conf)
            for l in sorted(page.links, key=lambda x: scores.get(x.href, 0), reverse=True)[:5]:
                if l.href not in self.visited:
                    heapq.heappush(heap, (-scores.get(l.href, 0), l.href, depth + 1))
        return {
            "goal": goal,
            "found": bool(best_answer.strip()),
            "confidence": best_conf,
            "answer": best_answer.strip(),
            "visited_pages": len(self.visited),
        }

    async def _evaluate_memory(self, goal: str):
        """
        Retrieve top-k relevant chunks from DB and ask the planner
        whether the GOAL is already satisfied.
        Returns (found: bool, confidence: float, answer: str).
        """
        if not self.db.has_embeddings(self.embed_model):
            return False, 0.0, ""

        # 1️⃣ Embed the goal itself
        q_vec = self.embedder.embed([goal])[0]

        # 2️⃣ Retrieve the most similar stored chunks
        retrieved = self.db.similarity_search(
            q_vec, top_k=self.retrieval_top_k, model=self.embed_model
        )
        if not retrieved:
            return False, 0.0, ""

        # 3️⃣ Concatenate retrieved context and let planner evaluate
        context_text = "\n\n".join(r["text"] for r in retrieved)
        found, confidence, answer = await self.planner.evaluate_context(goal, context_text)
        return found, confidence, answer


```

## `spiders/stealth/__init__.py`

```python

```

## `spiders/stealth/stealth_config.py`

```python
# spiders/stealth/stealth_config.py
# Configuration defaults for StealthSpider / VPN behavior.

DEFAULT_VPN_PROVIDER = "nordvpn"
DEFAULT_REGION = "hong_kong"
REQUIRE_VPN_DEFAULT = True

# Behavior toggles
DISCONNECT_BEFORE_LLM = True      # Disconnect VPN before making LLM API calls (recommended)
RECONNECT_AFTER_LLM = False       # Reconnect VPN after LLM calls (optional)
OBFUSCATE_BY_DEFAULT = True       # Try to enable obfuscation if provider supports it
PROTOCOL_DEFAULT = "tcp"          # prefer tcp for stealthy behaviour (OpenVPN over TCP/443)
CONNECT_TIMEOUT = 30              # seconds to wait for VPN connect
DISCONNECT_TIMEOUT = 10           # seconds to wait for VPN disconnect
MAX_CONNECT_RETRIES = 2

```

## `spiders/stealth/vpn_manager.py`

```python
# spiders/stealth/vpn_manager.py
"""
VPN helper for StealthSpider.

Currently supports NordVPN via the `nordvpn` CLI.
The manager:
 - can switch NordVPN "technology" to openvpn when required for obfuscation,
 - enable obfuscation,
 - attempt connect/disconnect with retries and timeouts,
 - queries the current connection state.

This is intentionally conservative in side-effects and logs clearly.
"""

import subprocess
import shlex
import time
from typing import Optional
import logging

logger = logging.getLogger("vpn_manager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[VPN] %(message)s"))
    logger.addHandler(ch)


class VPNError(Exception):
    pass


class VPNManager:
    def __init__(self, provider: str = "nordvpn"):
        self.provider = provider.lower()
        if self.provider != "nordvpn":
            raise VPNError("Only 'nordvpn' provider is implemented in this manager.")
        self._last_region = None

    def _run(self, cmd: str, timeout: int = 15) -> str:
        logger.debug(f"Running: {cmd}")
        parts = shlex.split(cmd)
        try:
            out = subprocess.check_output(parts, stderr=subprocess.STDOUT, timeout=timeout)
            return out.decode("utf-8", errors="replace")
        except subprocess.CalledProcessError as e:
            logger.debug(f"Cmd failed ({cmd}): {e.output.decode(errors='replace')}")
            raise VPNError(f"Command failed: {cmd}\n{e.output.decode(errors='replace')}")
        except subprocess.TimeoutExpired as e:
            logger.debug(f"Cmd timeout ({cmd})")
            raise VPNError(f"Command timeout: {cmd}")

    # ---------- NordVPN-specific helpers ----------
    def _nordvpn_status(self) -> str:
        return self._run("nordvpn status", timeout=6)

    def _is_connected(self) -> bool:
        try:
            out = self._nordvpn_status().lower()
            return "connected" in out
        except Exception:
            return False

    def _current_country(self) -> Optional[str]:
        try:
            out = self._nordvpn_status()
            # Status output contains "Country: Hong Kong" or "City: ..."
            for line in out.splitlines():
                if line.lower().startswith("country:"):
                    return line.split(":", 1)[1].strip().lower().replace(" ", "_")
        except Exception:
            return None

    def ensure_openvpn_for_obfuscation(self) -> None:
        """
        NordVPN disallows `obfuscate on` unless technology is set to openvpn.
        Switch to openvpn if needed.
        """
        try:
            tech_out = self._run("nordvpn settings")
            # quick check: if technology is already OpenVPN, bail
            if "Technology: OpenVPN" in tech_out or "technology: OpenVPN".lower() in tech_out.lower():
                logger.debug("NordVPN technology already OpenVPN.")
                return
        except Exception:
            # settings might vary; attempt to set openvpn anyway
            logger.debug("Could not read nordvpn settings; proceeding to set technology to openvpn.")

        logger.info("Setting NordVPN technology to OpenVPN (required for obfuscation).")
        try:
            self._run("nordvpn set technology openvpn", timeout=6)
        except Exception as e:
            logger.warning(f"Could not set technology to openvpn: {e}")

    def set_obfuscation(self, value: bool) -> None:
        val = "on" if value else "off"
        logger.info(f"Setting obfuscation: {val}")
        try:
            self._run(f"nordvpn set obfuscate {val}", timeout=6)
        except VPNError as e:
            # surfacing helpful message
            msg = str(e)
            if "not available" in msg or "not installed" in msg.lower():
                logger.warning("Obfuscation not supported with current settings/provider.")
            else:
                logger.warning(f"Failed to set obfuscation: {e}")

    def set_protocol(self, protocol: str) -> None:
        protocol = protocol.lower()
        if protocol not in ("tcp", "udp"):
            raise VPNError("protocol must be 'tcp' or 'udp'")
        logger.info(f"Setting NordVPN protocol to {protocol}")
        try:
            self._run(f"nordvpn set protocol {protocol}", timeout=6)
        except Exception as e:
            logger.warning(f"Failed to set protocol: {e}")

    def connect(self, region: str, obfuscate: bool = True, protocol: str = "tcp", timeout: int = 30) -> None:
        """
        Connect to provider's region. Uses retries and attempts to enable obfuscation if requested.
        """
        region = region.replace(" ", "_")
        self._last_region = region

        # ensure protocol + obfuscation readiness
        if obfuscate:
            # NordVPN requires openvpn tech for obfuscation
            try:
                self.ensure_openvpn_for_obfuscation()
            except Exception:
                logger.debug("ensure_openvpn_for_obfuscation had an issue, continuing.")

        # set protocol
        try:
            self.set_protocol(protocol)
        except Exception:
            logger.debug("set_protocol failed; continuing.")

        # enable obfuscation if requested
        if obfuscate:
            try:
                self.set_obfuscation(True)
            except Exception:
                logger.debug("set_obfuscation failed; continuing.")

        # attempt connect with retries
        last_err = None
        for attempt in range(1, 1 + 2):
            try:
                logger.info(f"Attempting VPN connection to: {region} (attempt {attempt})")
                out = self._run(f"nordvpn connect {region}", timeout=timeout)
                logger.info(out.strip().splitlines()[-1] if out else "Connected (no output)")
                # confirm connected
                start = time.time()
                while time.time() - start < (timeout if timeout else 30):
                    if self._is_connected():
                        logger.info(f"VPN connected to {region}")
                        return
                    time.sleep(1)
                raise VPNError("Connection attempt timed out")
            except VPNError as e:
                last_err = e
                logger.warning(f"Connect attempt failed: {e}")
                # try a fallback: turn obfuscation off and try TCP again if obfuscate was on
                if obfuscate:
                    try:
                        logger.info("Falling back: disabling obfuscation and retrying.")
                        self.set_obfuscation(False)
                    except Exception:
                        pass
                time.sleep(1)

        raise VPNError(f"Failed to connect to VPN {region}. Last error: {last_err}")

    def disconnect(self, timeout: int = 10) -> None:
        """
        Disconnect vpn client.
        """
        if not self._is_connected():
            logger.info("VPN already disconnected.")
            return
        logger.info("Disconnecting VPN...")
        try:
            out = self._run("nordvpn disconnect", timeout=timeout)
            logger.info("Disconnected.")
            # small wait to ensure new interface state
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to cleanly disconnect VPN: {e}")

    def running_provider(self) -> str:
        return self.provider

    def last_region(self) -> Optional[str]:
        return self._last_region

```

## `spiders/stealth_spider.py`

```python
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

```

## `storage/__init__.py`

```python
"""
spider_core.storage
-------------------
Persistence layer for crawl results, chunks, and vector embeddings.
"""

from spider_core.storage.db import DB

__all__ = ["DB"]

```

## `storage/db.py`

```python
# storage/db.py
import sqlite3
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Any, List, Dict, Tuple

import numpy as np


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS pages(
  id INTEGER PRIMARY KEY,
  url TEXT UNIQUE,
  canonical TEXT,
  status INTEGER,
  fetched_at INTEGER,
  title TEXT,
  visible_text TEXT
);

CREATE TABLE IF NOT EXISTS links(
  id INTEGER PRIMARY KEY,
  from_url TEXT,
  to_url TEXT,
  anchor_text TEXT,
  rel TEXT,
  llm_score_est REAL DEFAULT 0.0,
  llm_score_final REAL DEFAULT 0.0,
  UNIQUE(from_url, to_url)
);

CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  page_url TEXT,
  chunk_id INTEGER,
  text TEXT,
  token_count INTEGER,
  UNIQUE(page_url, chunk_id)
);

-- Simple vector storage (float32 array as JSON; small, portable)
CREATE TABLE IF NOT EXISTS embeddings(
  id INTEGER PRIMARY KEY,
  page_url TEXT,
  chunk_id INTEGER,
  vector TEXT,             -- json.dumps(list of floats)
  model TEXT,
  dim INTEGER,
  created_at INTEGER,
  UNIQUE(page_url, chunk_id, model)
);

CREATE TABLE IF NOT EXISTS crawl_log(
  id INTEGER PRIMARY KEY,
  url TEXT,
  action TEXT,             -- queued, fetched, skipped, failed
  reason TEXT,
  ts INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
CREATE INDEX IF NOT EXISTS idx_links_to ON links(to_url);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_url);
CREATE INDEX IF NOT EXISTS idx_embeds_page ON embeddings(page_url);
"""


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


class DB:
    """
    SQLite-backed storage for pages, links, chunks, and embeddings.

    New functionality:
      - has_embeddings(): quick check whether any embeddings exist (optionally by model).
      - similarity_search(): naive cosine similarity over all stored embeddings, joined with chunks.
    """

    def __init__(self, path: str = "spider_core.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Page / link / chunk persistence
    # ------------------------------------------------------------------
    def upsert_page(
        self,
        url: str,
        canonical: Optional[str],
        status: int,
        title: Optional[str],
        visible_text: str,
    ):
        self.conn.execute(
            """INSERT INTO pages(url, canonical, status, fetched_at, title, visible_text)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 canonical=excluded.canonical,
                 status=excluded.status,
                 fetched_at=excluded.fetched_at,
                 title=excluded.title,
                 visible_text=excluded.visible_text
            """,
            (url, canonical, status, int(time.time()), title, visible_text),
        )
        self.conn.commit()

    def upsert_links(self, from_url: str, links: Iterable[dict]):
        rows = []
        for l in links:
            rows.append(
                (
                    from_url,
                    l["href"],
                    l.get("text"),
                    json.dumps(l.get("rel", [])),
                    float(l.get("llm_score", 0.0)),
                )
            )
        self.conn.executemany(
            """INSERT INTO links(from_url, to_url, anchor_text, rel, llm_score_est)
               VALUES(?,?,?,?,?)
               ON CONFLICT(from_url,to_url) DO UPDATE SET
                 anchor_text=excluded.anchor_text,
                 rel=excluded.rel,
                 llm_score_est=excluded.llm_score_est
            """,
            rows,
        )
        self.conn.commit()

    def set_final_link_score(self, from_url: str, to_url: str, score: float):
        self.conn.execute(
            "UPDATE links SET llm_score_final=? WHERE from_url=? AND to_url=?",
            (float(score), from_url, to_url),
        )
        self.conn.commit()

    def upsert_chunks(self, page_url: str, chunks: Iterable[dict]):
        rows = []
        for c in chunks:
            rows.append(
                (
                    page_url,
                    int(c["chunk_id"]),
                    c["text"],
                    int(c["token_count"]),
                )
            )
        self.conn.executemany(
            """INSERT INTO chunks(page_url, chunk_id, text, token_count)
               VALUES(?,?,?,?)
               ON CONFLICT(page_url,chunk_id) DO UPDATE SET
                 text=excluded.text,
                 token_count=excluded.token_count
            """,
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Embeddings / RAG
    # ------------------------------------------------------------------
    def upsert_embedding(
        self,
        page_url: str,
        chunk_id: int,
        vec: List[float],
        model: str,
        dim: int,
    ):
        self.conn.execute(
            """INSERT INTO embeddings(page_url,chunk_id,vector,model,dim,created_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(page_url,chunk_id,model) DO UPDATE SET
                 vector=excluded.vector,
                 dim=excluded.dim,
                 created_at=excluded.created_at
            """,
            (page_url, chunk_id, json.dumps(vec), model, dim, int(time.time())),
        )
        self.conn.commit()

    def has_embeddings(self, model: Optional[str] = None) -> bool:
        cur = self.conn.cursor()
        if model:
            row = cur.execute(
                "SELECT 1 FROM embeddings WHERE model=? LIMIT 1",
                (model,),
            ).fetchone()
        else:
            row = cur.execute("SELECT 1 FROM embeddings LIMIT 1").fetchone()
        return row is not None

    def similarity_search(
        self,
        query_vec: List[float],
        top_k: int = 5,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Naive in-memory cosine similarity search over all embeddings
        (optionally filtered by model). Returns list of:
          {page_url, chunk_id, text, score}
        """
        cur = self.conn.cursor()
        if model:
            rows = cur.execute(
                "SELECT page_url, chunk_id, vector, dim FROM embeddings WHERE model=?",
                (model,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT page_url, chunk_id, vector, dim FROM embeddings",
            ).fetchall()

        if not rows:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        sims: List[Tuple[float, str, int]] = []

        for page_url, chunk_id, vec_json, dim in rows:
            try:
                v_list = json.loads(vec_json)
                v = np.asarray(v_list, dtype=np.float32)
            except Exception:
                continue
            if v.shape[0] != dim:
                continue
            score = _cosine_sim(q, v)
            sims.append((score, page_url, int(chunk_id)))

        sims.sort(key=lambda x: x[0], reverse=True)
        sims = sims[:top_k]

        # Fetch text for those chunks
        results: List[Dict[str, Any]] = []
        for score, page_url, chunk_id in sims:
            row = cur.execute(
                "SELECT text FROM chunks WHERE page_url=? AND chunk_id=?",
                (page_url, chunk_id),
            ).fetchone()
            text = row[0] if row else ""
            results.append(
                {
                    "page_url": page_url,
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": float(score),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def already_fetched(self, url: str) -> bool:
        r = self.conn.execute(
            "SELECT 1 FROM pages WHERE url=? LIMIT 1",
            (url,),
        ).fetchone()
        return r is not None

    def log(self, url: str, action: str, reason: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO crawl_log(url,action,reason,ts) VALUES(?,?,?,?)",
            (url, action, reason, int(time.time())),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

```

## `test/test.py`

```python
import asyncio
from browser.playwright_client import PlaywrightBrowserClient

async def test():
    client = PlaywrightBrowserClient()
    html, text, status = await client.render("https://nytimes.com")
    print("Status:", status)
    print("HTML length:", len(html))
    print("Visible text:", text[:200])
    await client.close()

asyncio.run(test())

```

## `test/test2.py`

```python
from extractors.deterministic_extractor import DeterministicLinkExtractor

sample_html = """
<a href="/about">About Us</a>
<link rel="canonical" href="https://example.com/home" />
<meta property="og:url" content="https://example.com/page" />
<div data-href="/contact">Get in Touch</div>
"""

links = DeterministicLinkExtractor.extract(sample_html, "https://example.com")
for link in links:
    print(link)

```

## `test/test3.py`

```python
from core_utils.chunking import TextChunker

text = """
This is a paragraph.
Another one.
And a very long paragraph that keeps going and might exceed a chunk limit depending on the token count, so this is just for demonstration purposes.
"""
chunker = TextChunker(model="gpt-4o-mini", max_tokens=20)
chunks = chunker.chunk_text(text)
for c in chunks:
    print(c)

```

## `test/test4.py`

```python
from llm.openai_gpt_client import OpenAIGPTClient
import asyncio

async def main():
    llm = OpenAIGPTClient()
    result = await llm.complete_json(
        "You are a JSON bot. Output only valid JSON with one key 'greet'.",
        "Say hi in JSON."
    )
    print(result)

asyncio.run(main())

```

## `test/test_ranker.py`

```python
import asyncio
from llm.openai_gpt_client import OpenAIGPTClient
from llm.relevance_ranker import RelevanceRanker
from base.link_metadata import LinkMetadata

async def main():
    llm = OpenAIGPTClient()
    ranker = RelevanceRanker(llm)

    links = [
        LinkMetadata(href="https://example.com/about", text="About Us", rel=[], detected_from=["a"]),
        LinkMetadata(href="https://example.com/contact", text="Contact", rel=[], detected_from=["a"]),
    ]

    chunks = [{"chunk_id": 0, "text": "This page discusses who we are and our company mission.", "token_count": 12}]

    enriched = await ranker.score_links(links, chunks)
    for link in enriched:
        print(link)

asyncio.run(main())

```

## `test/test_spider.py`

```python
import asyncio
from browser.playwright_client import PlaywrightBrowserClient
from core_utils.chunking import TextChunker
from llm.openai_gpt_client import OpenAIGPTClient
from llm.relevance_ranker import RelevanceRanker
from spiders.basic_spider import BasicSpider


async def main():
    browser = PlaywrightBrowserClient()
    llm = OpenAIGPTClient()
    ranker = RelevanceRanker(llm)
    chunker = TextChunker()

    spider = BasicSpider(browser, ranker, chunker)
    result = await spider.fetch("https://example.com")

    print(result)
    await browser.close()

asyncio.run(main())

```

<details>
<summary>📁 Final Project Structure</summary>

```
📁 base/
    📄 __init__.py
    📄 link_metadata.py
    📄 page_result.py
    📄 spider.py
📁 browser/
    📄 __init__.py
    📄 browser_client.py
    📄 playwright_client.py
📁 core_utils/
    📄 __init__.py
    📄 chunking.py
    📄 url_utils.py
📁 extractors/
    📄 __init__.py
    📄 deterministic_extractor.py
📁 goal/
    📄 __init__.py
    📄 goal_planner.py
📁 llm/
    📄 __init__.py
    📄 embeddings_client.py
    📄 llm_client.py
    📄 openai_gpt_client.py
    📄 relevance_ranker.py
📁 spiders/
    📁 stealth/
        📄 __init__.py
        📄 stealth_config.py
        📄 vpn_manager.py
    📄 __init__.py
    📄 basic_spider.py
    📄 goal_spider.py
    📄 stealth_spider.py
📁 storage/
    📄 __init__.py
    📄 db.py
📁 test/
    📄 test.py
    📄 test2.py
    📄 test3.py
    📄 test4.py
    📄 test_ranker.py
    📄 test_spider.py
📄 __init__.py
📄 cli_spider.py
📄 README.md
📄 requirements.txt
```

</details>
