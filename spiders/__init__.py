"""
spider_core.spiders
-------------------
All spider implementations (Basic, Stealth, Goal).
"""

from spider_core.spiders.basic_spider import BasicSpider
from spider_core.spiders.stealth_spider import StealthSpider
from spider_core.spiders.goal_spider import GoalSpider

__all__ = ["BasicSpider", "StealthSpider", "GoalSpider"]
