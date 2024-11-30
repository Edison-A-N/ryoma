from typing import List, Dict
import aiohttp
from bs4 import BeautifulSoup


class HTMLParser:
    """Simple HTML content parser."""

    async def _fetch_html(self, url: str) -> Dict[str, str]:
        """Fetch and extract content from URL.

        Args:
            url: Target URL

        Returns:
            Dict containing parsed content
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    html = await response.text()

                    if not html.strip():
                        raise ValueError(f"Empty response from {url}")

                    soup = BeautifulSoup(html, "html.parser")

                    # Remove scripts and styles
                    for tag in soup.find_all(["script", "style"]):
                        tag.decompose()

                    return {
                        "title": soup.title.text.strip() if soup.title else "",
                        "content": soup.get_text(separator="\n", strip=True),
                    }

        except Exception as e:
            raise ValueError(f"Failed to parse {url}: {str(e)}")

    async def parse_url(self, url: str) -> Dict[str, str]:
        """Parse content from URL.

        Args:
            url: URL to fetch and parse

        Returns:
            Dict containing parsed content
        """
        return await self._fetch_html(url)

    async def parse_batch(self, urls: List[str]) -> List[Dict[str, str]]:
        """Parse multiple URLs.

        Args:
            urls: List of URLs to fetch and parse

        Returns:
            List of parsed content dictionaries
        """
        return [await self._fetch_html(url) for url in urls]
