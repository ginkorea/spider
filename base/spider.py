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
