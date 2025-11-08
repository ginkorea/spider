"""
spider_core.core_utils
----------------------
Utility functions for URL normalization, chunking, and tokenization.
"""

from spider_core.core_utils.chunking import TextChunker
from spider_core.core_utils.url_utils import canonicalize_url

__all__ = ["TextChunker", "canonicalize_url"]
