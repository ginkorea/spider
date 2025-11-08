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
