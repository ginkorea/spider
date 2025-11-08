"""
spider_core.storage
-------------------
Persistence layer for crawl results, chunks, and vector embeddings.
"""

from spider_core.storage.db import DB

__all__ = ["DB"]
