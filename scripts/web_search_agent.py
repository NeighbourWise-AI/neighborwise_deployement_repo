"""
web_search_agent.py — NeighbourWise AI
Web search agent: Serper (web + news) → Claude draft → Universal Validator.

Renamed from WebsearchAgent/app.py — now lives in scripts/ alongside all other agents.

Standalone test:
    python3 web_search_agent.py

Via router:
    from web_search_agent import serper_search, format_web_results,
                                  format_news_results, deep_fetch_top_urls,
                                  generate_draft
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# ── Universal Validator ────────────────────────────────────────────────────────
from universal_validator import UniversalValidator, AgentType
from shared.snowflake_conn import get_conn

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
SERPER_API_KEY   = os.environ.get("SERPER_API_KEY")


# ── Claude retry wrapper ───────────────────────────────────────────────────────
def call_claude_with_retry(client, max_retries=3, wait_seconds=15, **kwargs):
    """Retry Claude API calls on 529 Overloaded errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.OverloadedError:
            if attempt < max_retries:
                wait = wait_seconds * attempt
                print(f"Anthropic overloaded — retry in {wait}s ({attempt}/{max_retries})")
                time.sleep(wait)
            else:
                raise


# ── Serper helpers (imported by router_agent.py) ───────────────────────────────
def serper_search(query: str, search_type: str = "search", num_results: int = 10) -> dict:
    url     = f"https://google.serper.dev/{search_type}"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results, "gl": "us", "hl": "en"}
    resp    = requests.post(url, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def format_web_results(data: dict, label: str = "WEB") -> str:
    lines = [f"=== {label} RESULTS ==="]
    if data.get("answerBox"):
        ab = data["answerBox"]
        lines.append("--- ANSWER BOX (top priority) ---")
        if ab.get("title"):   lines.append(f"Title: {ab['title']}")
        if ab.get("answer"):  lines.append(f"Answer: {ab['answer']}")
        if ab.get("snippet"): lines.append(f"Detail: {ab['snippet']}")
        if ab.get("snippetHighlighted"):
            lines.append(f"Key points: {'; '.join(ab['snippetHighlighted'])}")
        lines.append("")
    if data.get("topStories"):
        lines.append("--- TOP STORIES ---")
        for s in data["topStories"][:5]:
            lines.append(f"  [{s.get('date','')}] {s.get('title','')}")
            lines.append(f"  URL: {s.get('link','')}")
        lines.append("")
    if data.get("organic"):
        lines.append("--- ORGANIC ---")
        for i, r in enumerate(data["organic"], 1):
            lines.append(f"[{i}] {r.get('title','')}")
            lines.append(f"    URL: {r.get('link','')}")
            if r.get("date"):
                lines.append(f"    Date: {r['date']}")
            lines.append(f"    Snippet: {r.get('snippet','')}")
            lines.append("")
    if data.get("peopleAlsoAsk"):
        lines.append("--- PEOPLE ALSO ASK ---")
        for q in data["peopleAlsoAsk"][:3]:
            lines.append(f"Q: {q.get('question','')}")
            lines.append(f"A: {q.get('snippet','')}")
            lines.append("")
    return "\n".join(lines)


def format_news_results(data: dict) -> str:
    lines    = ["=== NEWS RESULTS ==="]
    articles = data.get("news", [])
    if not articles:
        return ""
    for i, a in enumerate(articles[:8], 1):
        lines.append(f"[N{i}] {a.get('title','')}")
        lines.append(f"     Source: {a.get('source','')}")
        lines.append(f"     Date: {a.get('date','')}")
        lines.append(f"     URL: {a.get('link','')}")
        lines.append(f"     Snippet: {a.get('snippet','')}")
        lines.append("")
    return "\n".join(lines)


# ── Domain-specific URL priorities ────────────────────────────────────────────
DOMAIN_URL_PRIORITIES = {
    "Restaurants":  ["eater.com", "bostonmagazine.com", "theinfatuation.com",
                     "boston.com", "resy.com", "boston25", "wcvb.com"],
    "Housing":      ["bostonglobe.com", "bostonmagazine.com", "boston.com",
                     "mass.gov", "bostonhousing.org", "bisnow.com"],
    "Crime/Safety": ["boston.gov", "bpdnews.com", "nbcboston.com",
                     "wcvb.com", "cbsnews.com", "masslive.com"],
    "Healthcare":   ["bostonglobe.com", "nbcboston.com", "wcvb.com",
                     "bostonmagazine.com", "mass.gov", "bmc.org"],
    "Schools":      ["bostonglobe.com", "mass.gov", "bostonpublicschools.org",
                     "wbur.org", "nbcboston.com"],
    "Grocery":      ["bostonglobe.com", "boston.com", "masslive.com",
                     "nbcboston.com", "wcvb.com"],
    "MBTA":         ["mbta.com", "bostonglobe.com", "wbur.org",
                     "nbcboston.com", "masslive.com", "universalhub.com"],
    "Weather":      ["weather.gov", "bostonglobe.com", "wcvb.com",
                     "nbcboston.com", "wbur.org", "masslive.com"],
}

BLOCKED_FETCH = [
    "tripadvisor.com", "yelp.com", "reddit.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "youtube.com",
    "linkedin.com", "pinterest.com", "tiktok.com",
]


def deep_fetch_top_urls(
    organic_results: list, domain: str, max_fetch: int = 3
) -> str:
    """
    Fetch full text from the most relevant organic URLs for the domain.
    Prioritizes domain-specific high-quality sources.
    Returns formatted string to append to search context.
    """
    import re
    priority_domains = DOMAIN_URL_PRIORITIES.get(domain, [])

    scored = []
    for r in organic_results:
        url = r.get("link", "")
        if any(b in url for b in BLOCKED_FETCH):
            continue
        score = 0
        for i, pd in enumerate(priority_domains):
            if pd in url:
                score = len(priority_domains) - i
                break
        scored.append((score, url, r.get("title", "")))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_urls = [(url, title) for _, url, title in scored[:max_fetch]]

    if not top_urls:
        return ""

    fetched_lines = ["=== DEEP FETCHED CONTENT ==="]
    hdrs = {"User-Agent": "Mozilla/5.0 (compatible; NeighbourWiseBot/1.0)"}

    for url, title in top_urls:
        try:
            resp = requests.get(url, headers=hdrs, timeout=8)
            if resp.status_code != 200:
                continue
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()[:2500]
            fetched_lines.append(f"--- {title} ---")
            fetched_lines.append(f"URL: {url}")
            fetched_lines.append(f"Content: {text}")
            fetched_lines.append("")
            print(f"   Fetched: {url[:70]}...")
        except Exception as e:
            print(f"   Fetch failed {url[:60]}: {e}")

    return "\n".join(fetched_lines) if len(fetched_lines) > 1 else ""


# ── Claude draft generation ────────────────────────────────────────────────────
def generate_draft(query: str, domain: str, search_context: str) -> str:
    """
    Claude Opus generates the initial draft from Serper search context.
    Claude generates → GPT-4o (or Claude fallback) validates.
    Cross-model validation is intentional.
    """
    domain_ctx = (
        f'This query is about the "{domain}" domain of neighborhood livability.'
        if domain != "All" else "This query covers neighborhood livability."
    )
    system_prompt = f"""You are a neighborhood intelligence analyst for NeighbourWise AI, \
reporting on livability conditions in Boston, MA. {domain_ctx}

STRUCTURE YOUR RESPONSE AS FOLLOWS:
1. OVERVIEW PARAGRAPH (3-5 sentences) — summarize overall situation with context. Cite [N].
2. KEY INCIDENTS & ALERTS (one ## section per distinct item) — exact date, time, \
   street address, what happened, status/outcome. Cite [N].
3. BACKGROUND & CONTEXT — statistics, resources, how to stay informed. Cite [N].
4. SOURCES — numbered URL list.

RULES:
- Every specific fact (date, address, number, name) must match sources exactly.
- Every factual sentence must end with [N] citation.
- No markdown bold (**text**). Use ## headings only.
- Target 400-600 words."""

    resp = call_claude_with_retry(
        anthropic_client,
        model="claude-opus-4-5",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content":
            f"Query: {query}\n\n--- SEARCH RESULTS ---\n{search_context}\n---\n\nWrite your response."}]
    )
    return resp.content[0].text.strip()


# ── Domain news keyword map (shared with router) ───────────────────────────────
DOMAIN_NEWS_KEYWORDS = {
    "Crime/Safety":  "crime safety alert incident police",
    "Housing":       "housing rent apartments development",
    "Restaurants":   "restaurant opening closing dining food",
    "Healthcare":    "hospital clinic healthcare medical",
    "Schools":       "school education MCAS district",
    "Grocery":       "grocery store supermarket food market",
    "MBTA":          "MBTA transit bus train service",
    "Weather":       "weather storm flood snow Boston",
}


# ── Main search function (called by router_agent.py) ──────────────────────────
def run_web_search(query: str, domain: str = "All", use_validator: bool = True) -> dict:
    """
    Full web search pipeline:
      1. Serper web + news search
      2. Deep fetch top URLs
      3. Claude Opus draft
      4. Universal Validator (GPT-4o if available, Claude fallback)

    Returns:
    {
        "answer":          str,
        "validation":      dict | None,
        "sources_fetched": int,
    }
    """
    if not SERPER_API_KEY:
        return {"answer": "", "error": "SERPER_API_KEY not set", "sources_fetched": 0}

    news_suffix = DOMAIN_NEWS_KEYWORDS.get(domain, "Boston 2026")

    # Step 1: Serper search
    try:
        web_data  = serper_search(query, search_type="search", num_results=10)
        news_data = serper_search(
            query + f" {news_suffix} 2025 2026",
            search_type="news", num_results=8,
        )
        organic_count = len(web_data.get("organic", []))
        search_ctx    = (
            format_web_results(web_data) + "\n\n" +
            format_news_results(news_data)
        )
    except requests.HTTPError as e:
        return {"answer": "", "error": f"Serper error: {e.response.status_code}", "sources_fetched": 0}
    except requests.RequestException as e:
        return {"answer": "", "error": f"Serper failed: {str(e)}", "sources_fetched": 0}

    # Step 1b: Deep fetch
    try:
        fetched = deep_fetch_top_urls(web_data.get("organic", []), domain, max_fetch=3)
        if fetched:
            search_ctx += "\n\n" + fetched
    except Exception as e:
        print(f"[WebSearch] Deep fetch warning (non-fatal): {e}")

    # Step 2: Claude draft
    try:
        draft = generate_draft(query, domain, search_ctx)
        print(f"[WebSearch] Draft generated ({len(draft)} chars)")
    except Exception as e:
        return {"answer": "", "error": f"Claude draft error: {str(e)}", "sources_fetched": organic_count}

    # Step 3: Validate via UniversalValidator
    final_output    = draft
    validation_meta = None

    if use_validator:
        conn = None
        try:
            conn      = get_conn()
            validator = UniversalValidator(conn)
            val_result = validator.validate(
                AgentType.WEB_SEARCH,
                {
                    "query":          query,
                    "domain":         domain,
                    "draft":          draft,
                    "search_context": search_ctx,
                }
            )
            val_result.print_summary()
            final_output    = val_result.result if val_result.result else draft
            validation_meta = {
                "passed":   val_result.passed,
                "improved": val_result.improved,
                "issues":   val_result.all_issues,
                "score": next(
                    (
                        c.details.get("score")
                        for c in val_result.checks.values()
                        if c.details.get("score") is not None
                    ),
                    None,
                ),
            }
        except Exception as e:
            print(f"[WebSearch] Validation error (non-fatal): {e}")
            validation_meta = {"error": str(e)}
        finally:
            if conn:
                conn.close()

    return {
        "answer":          final_output,
        "validation":      validation_meta,
        "sources_fetched": organic_count,
    }


# ── Standalone test (terminal) ─────────────────────────────────────────────────
def main():
    missing = [k for k in ["ANTHROPIC_API_KEY", "SERPER_API_KEY"]
               if not os.environ.get(k)]
    if missing:
        print(f"\nERROR: Missing environment variables: {', '.join(missing)}")
        for k in missing:
            print(f"  export {k}=your_key_here")
        sys.exit(1)

    print("\n NeighbourWise Web Search Agent (standalone)")
    print(" Claude Opus generates → GPT-4o (or Claude) validates")
    print(" Type 'exit' to quit\n")

    while True:
        try:
            query = input("  Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query or query.lower() in ("exit", "quit"):
            break

        domain = input("  Domain [All]: ").strip() or "All"
        result = run_web_search(query, domain=domain)

        print(f"\n{'─'*65}")
        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        else:
            print(result["answer"])
            val = result.get("validation")
            if val:
                score   = val.get("score", "?")
                passed  = val.get("passed", False)
                improved = val.get("improved", False)
                tag     = "🔍 Improved" if improved else ("✅ Passed" if passed else "❌ Failed")
                print(f"\n🛡️  Validation: {tag} | GPT-4o score: {score}/100")
        print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()