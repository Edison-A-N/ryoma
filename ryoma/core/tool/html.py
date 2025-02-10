from typing import List, Dict
from bs4 import BeautifulSoup
import random
import requests

from ryoma.core.logging import logger


class HTMLParser:
    """Simple HTML content parser."""

    async def _fetch_html(self, url: str) -> Dict[str, str]:
        """Fetch and extract content from URL.

        Args:
            url: Target URL

        Returns:
            Dict containing parsed content
        """
        headers = {
            "User-Agent": random.choice(
                [
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36",
                ]
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=10,
                allow_redirects=True,
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            html = response.text

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
            logger.error(f"Failed to fetch HTML from {url}: {str(e)}")
            raise

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
