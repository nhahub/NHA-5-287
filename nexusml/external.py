"""External open web context via the Tavily Search API (free tier: 1,000 credits/month; 1 credit per basic search).

OPTIONAL: set TAVILY_API_KEY to enable. The key is read here only and never put in state, prompts or reports.
Only public-context queries written by the LLM are sent (never dataset values); the queries are listed in the report."""
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List
from urllib.parse import urlparse

URL = "https://api.tavily.com/search"


class ExternalError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def search(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    if not available():
        raise ExternalError("TAVILY_API_KEY is not set")
    req = urllib.request.Request(URL, method="POST", data=json.dumps({"query": query[:300], "search_depth": "basic", "max_results": max_results,
                                                                      "include_answer": False}).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["TAVILY_API_KEY"],
                                          "User-Agent": "nexusml/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ExternalError(f"Tavily HTTP {e.code}" + (" (check TAVILY_API_KEY)" if e.code in (401, 403) else " (rate/credit limit?)" if e.code == 429 else ""))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ExternalError(f"Tavily network error ({type(e).__name__})")
    out = []
    for x in data.get("results", [])[:max_results]:
        out.append({"title": str(x.get("title", ""))[:140], "url": str(x.get("url", "")), "domain": urlparse(str(x.get("url", ""))).netloc.replace("www.", ""),
                    "snippet": " ".join(str(x.get("content", "")).split())[:220], "score": x.get("score")})
    return out
