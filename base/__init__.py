"""
spider_core.base
----------------
Base interfaces and data models used by all spiders.
"""

from spider_core.base.spider import Spider
from spider_core.base.page_result import PageResult
from spider_core.base.link_metadata import LinkMetadata

__all__ = ["Spider", "PageResult", "LinkMetadata"]
