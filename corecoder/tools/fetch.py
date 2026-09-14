"""A read-only tool that fetches the text content of a URL."""

import urllib.request
from .base import Tool


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch the text content of an http(s) URL. "
        "Use this to read documentation, API responses, or web pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The http:// or https:// URL to fetch",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 15)",
            },
        },
        "required": ["url"],
    }

    def execute(self, url: str, timeout: int = 15) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: only http and https URLs are supported"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CoreCoder"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(1_000_000)  # 顶多读 1MB，别让一个巨页撑爆内存
                text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error fetching {url}: {e}"
        # 太长就留头尾，跟 bash 输出截断一个套路
        if len(text) > 8000:
            text = text[:6000] + f"\n... (truncated, {len(text)} chars) ...\n" + text[-1000:]
        return text