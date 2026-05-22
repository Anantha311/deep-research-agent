import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
    domain: str = field(init=False)

    def __post_init__(self):
        from urllib.parse import urlparse
        self.domain = urlparse(self.url).netloc


def search(query: str, max_results) -> list[SearchResult]:
    client = TavilyClient(api_key=TAVILY_API_KEY)
    raw = client.search(query=query, search_depth="advanced",
                        max_results=max_results, include_answer=False)
    return [
        SearchResult(title=r.get("title","Untitled"), url=r.get("url",""),snippet=r.get("content",""), score=r.get("score"))
        for r in raw.get("results", [])
        ]
