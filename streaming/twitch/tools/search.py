"""
Search handler for Alice's web search tool.

Architecture:

    Alice fires search_web
        ↓
    [PRE-FILTER]  fairy.check_query()
        ↓
    [SEARCH]      Tavily API call
        ↓
    [POST-FILTER] fairy.check_results()
        ↓
    [SUMMARIZE]   compress to ~2-3 sentences for Alice's context
        ↓
    Returned to Alice

Fairy is pluggable — it runs both TOS filtering and the full security
scanner (prompt-injection guard, secret detection, suspicious URL filter). Pass a real implementation in when wiring; the default is
a safe no-op stub that passes everything through (so the handler runs
end-to-end during dev).

To plug in the real Fairy:

    handler = SearchHandler(
        api_key=os.environ["TAVILY_API_KEY"],
        fairy=YourFairySystem(),
        summarizer=your_summarizer_fn,
    )
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import httpx

log = logging.getLogger("alice.search")


# =============================================================================
# Filter contract (so the real Fairy can be swapped in cleanly)
# =============================================================================

@dataclass
class FilterDecision:
    """
    Returned by Fairy. Allow/deny + optional reason and
    optional sanitized version of the input.
    """
    allow: bool
    reason: str = ""
    # If a filter wants to scrub instead of block, it returns sanitized data.
    sanitized: Optional[Any] = None


class QueryFilter(Protocol):
    """Pre-search filter. Inspects Alice's query before it hits the API."""

    def check_query(self, query: str, intent: str) -> FilterDecision:
        ...


class ResultsFilter(Protocol):
    """Post-search filter. Inspects search results before Alice sees them."""

    def check_results(
        self, query: str, results: list[dict]
    ) -> FilterDecision:
        ...


class SafetyLayer(QueryFilter, ResultsFilter, Protocol):
    """A full filter implements both pre and post filtering."""
    pass


# =============================================================================
# Default no-op stubs (replace with the real Fairy in production)
# =============================================================================

class PassthroughFilter:
    """
    Default filter that allows everything. Used until Fairy is wired.
    NEVER ship to production with this as the only layer.
    """

    def check_query(self, query: str, intent: str) -> FilterDecision:
        return FilterDecision(allow=True, reason="passthrough")

    def check_results(
        self, query: str, results: list[dict]
    ) -> FilterDecision:
        return FilterDecision(allow=True, reason="passthrough")


class BasicFairyStub:
    """
    Minimal hardcoded filter as a placeholder for the real fairy system.
    Catches the obvious stuff so dev testing isn't completely unguarded.
    Real fairy system replaces this entirely.
    """

    # Hard-block patterns. Real fairy will be smarter, but this catches the
    # most basic categories of bad query.
    BLOCKED_QUERY_PATTERNS = [
        # Doxing attempts
        "home address",
        "phone number of",
        "social security",
        "ssn of",
        # Real-person sexual content
        "nude",
        "naked celebrity",
        # Obvious prompt injection attempts at the search layer
        "ignore previous instructions",
        "system prompt",
    ]

    BLOCKED_DOMAINS = {
        # Add domains as discovered. Stub list for dev.
        "4chan.org",
        "kiwifarms.net",
    }

    def check_query(self, query: str, intent: str) -> FilterDecision:
        q = query.lower()
        for pattern in self.BLOCKED_QUERY_PATTERNS:
            if pattern in q:
                return FilterDecision(
                    allow=False,
                    reason=f"fairy: query matched blocked pattern '{pattern}'",
                )
        return FilterDecision(allow=True, reason="fairy: query clean")

    def check_results(
        self, query: str, results: list[dict]
    ) -> FilterDecision:
        sanitized = []
        dropped = 0
        for r in results:
            url = r.get("url", "")
            if any(domain in url for domain in self.BLOCKED_DOMAINS):
                dropped += 1
                continue
            sanitized.append(r)

        if dropped > 0:
            return FilterDecision(
                allow=True,
                reason=f"fairy: dropped {dropped} result(s) from blocked domains",
                sanitized=sanitized,
            )
        return FilterDecision(allow=True, reason="fairy: results clean")


# =============================================================================
# Result types
# =============================================================================

@dataclass
class SearchResult:
    """A single search result, normalized across backends."""
    title: str
    url: str
    snippet: str
    score: float = 0.0          # backend-provided relevance, 0-1
    published_date: Optional[str] = None  # ISO date if available

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "published_date": self.published_date,
        }


@dataclass
class SearchResponse:
    """What gets returned from the handler back to the dispatcher."""
    query: str
    summary: str               # Compressed, Alice-readable
    results: list[SearchResult]
    sources_for_attribution: list[str]  # URLs Alice can mention
    duration_ms: float = 0.0
    blocked: bool = False
    blocked_reason: str = ""


# =============================================================================
# Tavily backend
# =============================================================================

class TavilyBackend:
    """
    Wrapper around Tavily's search API.
    https://docs.tavily.com

    Tavily was chosen because:
      - Built for LLM agents specifically
      - Returns structured, clean snippets (no scraping junk)
      - Has its own basic safety filtering as a third layer
      - Cheap (~$0.005/query)
      - Fast (~500ms p50)
    """

    BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str, timeout_seconds: float = 8.0):
        if not api_key:
            raise ValueError("Tavily API key required")
        self.api_key = api_key
        self.timeout = timeout_seconds

    async def search(
        self,
        query: str,
        max_results: int = 5,
        freshness: str = "any",
    ) -> list[SearchResult]:
        # Tavily's freshness param: 'day', 'week', 'month', 'year' or null
        time_range = None if freshness == "any" else freshness

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",   # 'advanced' costs more, slower
            "include_answer": True,    # Tavily generates a quick summary
            "include_raw_content": False,
        }
        if time_range:
            payload["time_range"] = time_range

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        results = []
        for r in data.get("results", []):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score", 0.0),
                    published_date=r.get("published_date"),
                )
            )

        # Tavily's auto-summary, if available, is tucked into 'answer'.
        # We surface it so the summarizer can use it as a starting point.
        tavily_answer = data.get("answer", "")
        if tavily_answer:
            # Stash on the first result for the summarizer to access.
            # Slight hack but avoids changing the dataclass.
            if results:
                results[0].snippet = (
                    f"[tavily_summary] {tavily_answer}\n\n{results[0].snippet}"
                )

        return results


# =============================================================================
# Summarizer (default = simple template; replace with auxiliary model call in prod)
# =============================================================================

SummarizerFn = Callable[[str, list[SearchResult]], str]


def default_summarizer(query: str, results: list[SearchResult]) -> str:
    """
    Default template-based summarizer. No LLM call.
    Replace with a function that calls Alice's auxiliary thoughts model on a
    separate CUDA stream for proper natural-language summarization.
    """
    if not results:
        return f"Searched for '{query}' but got nothing useful."

    # Check if Tavily gave us its own answer (stashed on first result)
    first = results[0]
    if first.snippet.startswith("[tavily_summary]"):
        try:
            answer = first.snippet.split("\n\n", 1)[0].replace(
                "[tavily_summary] ", ""
            )
            sources = ", ".join(
                _domain(r.url) for r in results[:3] if r.url
            )
            return f"{answer} (sources: {sources})"
        except Exception:
            pass

    # Fallback: stitch top 3 snippets
    snippets = []
    for r in results[:3]:
        if r.snippet and not r.snippet.startswith("[tavily_summary]"):
            snippets.append(f"{_domain(r.url)}: {r.snippet[:150]}")

    if not snippets:
        return f"Found {len(results)} results for '{query}' but they're sparse."

    return " | ".join(snippets)


def _domain(url: str) -> str:
    """Extract a clean domain name for source attribution."""
    if not url:
        return "unknown"
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "") if netloc else "unknown"
    except Exception:
        return "unknown"


# =============================================================================
# Main handler
# =============================================================================

class SearchHandler:
    """
    The full search pipeline. Plugs into the dispatcher as the handler for
    the search_web tool.

    Wiring example:

        handler = SearchHandler(
            api_key=os.environ["TAVILY_API_KEY"],
            fairy=fairy_system,    # the real unified Fairy (TOS + security)
            summarizer=alice_summarize_with_thoughts_llm,
        )
        dispatcher.register("search_web", handler.handle)
    """

    def __init__(
        self,
        api_key: str,
        fairy: Optional[SafetyLayer] = None,
        summarizer: Optional[SummarizerFn] = None,
        max_results: int = 5,
    ):
        self.backend = TavilyBackend(api_key=api_key)
        self.fairy = fairy or BasicFairyStub()
        self.summarize = summarizer or default_summarizer
        self.max_results = max_results

    async def handle(self, args: dict) -> SearchResponse:
        """Dispatcher-facing entry point. Args come pre-validated."""
        start = time.time()
        query = args["query"]
        intent = args["intent"]
        freshness = args.get("freshness", "any")

        # ---------- PRE-FILTER: fairy ----------
        fairy_pre = self.fairy.check_query(query, intent)
        if not fairy_pre.allow:
            log.warning(f"fairy blocked query: {query!r} - {fairy_pre.reason}")
            return SearchResponse(
                query=query,
                summary=(
                    "Couldn't search that - it tripped a content filter. "
                    "Try rephrasing."
                ),
                results=[],
                sources_for_attribution=[],
                duration_ms=(time.time() - start) * 1000,
                blocked=True,
                blocked_reason=fairy_pre.reason,
            )

        # ---------- SEARCH ----------
        try:
            results = await self.backend.search(
                query=query,
                max_results=self.max_results,
                freshness=freshness,
            )
        except Exception as e:
            log.exception(f"Search backend failed for query: {query!r}")
            return SearchResponse(
                query=query,
                summary="Search broke. Skip it for now.",
                results=[],
                sources_for_attribution=[],
                duration_ms=(time.time() - start) * 1000,
                blocked=False,
                blocked_reason=f"backend_error: {e}",
            )

        # ---------- POST-FILTER: fairy ----------
        result_dicts = [r.to_dict() for r in results]
        fairy_post = self.fairy.check_results(query, result_dicts)
        if not fairy_post.allow:
            log.warning(f"fairy blocked results: {fairy_post.reason}")
            return SearchResponse(
                query=query,
                summary="Results came back dirty - skipping.",
                results=[],
                sources_for_attribution=[],
                duration_ms=(time.time() - start) * 1000,
                blocked=True,
                blocked_reason=fairy_post.reason,
            )
        if fairy_post.sanitized is not None:
            results = [SearchResult(**d) for d in fairy_post.sanitized]

        # ---------- SUMMARIZE ----------
        summary = self.summarize(query, results)
        sources = [r.url for r in results[:3] if r.url]

        return SearchResponse(
            query=query,
            summary=summary,
            results=results,
            sources_for_attribution=sources,
            duration_ms=(time.time() - start) * 1000,
            blocked=False,
        )


# =============================================================================
# Adapter for dispatcher format
# =============================================================================

async def search_handler_for_dispatcher(
    args: dict,
    handler: SearchHandler,
) -> str:
    """
    Wrap SearchHandler.handle to return what the dispatcher's success
    formatter expects. The dispatcher's _format_success doesn't have a
    branch for search_web yet - this returns a string the dispatcher can
    use directly as the message.
    """
    response = await handler.handle(args)
    if response.blocked:
        return response.summary
    return response.summary
