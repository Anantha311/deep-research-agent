import trafilatura
from dataclasses import dataclass
from urllib.parse import urlparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional


@dataclass
class PageContent:
    url: str
    title: str
    content: str
    retrieved_at: str
    domain: str
    score: Optional[float] = None

def fetch_pages(url: str) -> Optional[PageContent]:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    text = trafilatura.extract(downloaded)
    if text is None:
        return None
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata else "Unknown" 
    domain = urlparse(url).netloc 
    return PageContent(url=url,title=title,content=text,retrieved_at=datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),domain=domain)

def default_page(url: str) -> PageContent:

    return PageContent(
        
        url=url,
        title="Unavailable Source",
        content="Content could not be retrieved from this webpage.",
        retrieved_at=datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        domain=urlparse(url).netloc
    )
