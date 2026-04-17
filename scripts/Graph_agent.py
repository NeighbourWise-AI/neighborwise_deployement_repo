"""
Graph_agent.py  (v4.3 — fix validator rank position verification)
=========================================================
NeighbourWise AI — Graph Agent

What changed from v4.2:
  validate_node() was passing detail metrics but NOT rank positions
  (e.g. "ranks 29 out of 51") to the validator context.  These rank summaries
  come from neo4j_neighborhood_rank() and are injected into the synthesize
  prompt, so Claude correctly cites them — but GPT-4o couldn't find them in
  the validator ground truth and flagged them as fabricated data.

  Fix: inject rank summaries into verified_detail_metrics alongside the
  detail metric lines, so GPT-4o sees:
    BACK BAY Safety RANK: BACK BAY ranks 29 out of 51 neighborhoods ...
    BACK BAY Safety: total_incidents=10449, violent_crime_count=119

What changed from v4.1 → v4.2:
  validate_node() was passing domain_metrics as a raw list of dicts under the
  key "domain_metrics", but _build_graph_validation_context() in
  universal_validator.py looks for the key "verified_detail_metrics" (a
  pre-formatted string). This mismatch meant GPT-4o never saw the detail
  figures (total_incidents, avg_price_per_sqft, etc.) in a labeled,
  neighborhood-attributed format — causing it to flag correctly-cited numbers
  as fabricated.

  Fix: pre-format domain_metrics into labeled sentences
  (e.g. "BACK BAY Safety: total_incidents=10449, violent_crime_count=119")
  and pass them under the correct key "verified_detail_metrics".

What changed from v4 → v4.1:
  Neighbourhood extraction was the root cause of all "data not found" failures.
  "Backbay" / "BackBay" / "back bay" / "Southie" / "JP" / "Eastie" all now
  resolve correctly to their canonical names before any DB call is made.

  Three-layer fix in extract_neighborhood() / extract_all_neighborhoods():
    1. Alias map — nospace variants ("BACKBAY"→"BACK BAY") + common nicknames
       ("SOUTHIE"→"SOUTH BOSTON", "JP"→"JAMAICA PLAIN", "EASTIE"→"EAST BOSTON")
    2. Subsumption pruning — removes redundant sub-names when a longer match
       exists ("ROXBURY" dropped when "WEST ROXBURY" is also matched,
       "BOSTON" dropped when "EAST BOSTON" / "DOWNTOWN" is matched)
    3. Meta-question guard — "best neighborhood in Boston" returns [] instead
       of matching "BOSTON" as a specific neighborhood query

  Multi-neighbourhood queries ("Allston vs Brighton") return all matches
  ordered by position in the query string (first-mentioned = primary).

  Meta-question handling ("best neighborhood in Boston"):
  extract_all_neighborhoods() correctly returns [] (no specific neighbourhood),
  so plan_node sets neighborhood=None. The neo4j_node and mart_node then skip
  the single-neighbourhood profile queries and only run top_by_domain — returning
  the ranked list across all 51 neighbourhoods, which is exactly what the query needs.

LangGraph structure (unchanged from v4):
  [plan] → Send("neo4j_node") ──┐
         → Send("mart_node")  ──┼→ [merge] → [synthesize] → [validate] → END
         → Send("rag_node")   ──┘

Usage (unchanged):
    python Graph_agent.py -q "Is Backbay safe and affordable?"
    python Graph_agent.py -q "Best transit access" -n CAMBRIDGE
    python Graph_agent.py -i
    from Graph_agent import ask_graph_agent
"""

import os
import re
import sys
import json
import time
import textwrap
import argparse
import logging
import operator
from pathlib import Path
from typing import Optional, Annotated

from dotenv import load_dotenv
from neo4j import GraphDatabase
import anthropic

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing_extensions import TypedDict

from universal_validator import UniversalValidator, AgentType, validate_graph_output


# ── Env ───────────────────────────────────────────────────────────────────────

def _find_env_file() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        current = current.parent
    return Path(".env")

_env_path = _find_env_file()
load_dotenv(dotenv_path=_env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("graph_agent")


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"\n  Missing required env variable: {key}\n"
            f"  Set it in your .env file at: {_env_path}\n"
        )
    return val


# ── Credentials ───────────────────────────────────────────────────────────────

NEO4J_URI      = _require("NEO4J_URI")
NEO4J_USER     = _require("NEO4J_USERNAME")
NEO4J_PASSWORD = _require("NEO4J_PASSWORD")

# Snowflake credentials removed from graph agent.
# This agent is Neo4j-only. Snowflake access lives in the data_query / Cortex agent.

ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = "claude-sonnet-4-5"

# RAG domain property map — matches property names written by neo4j_schema_loader.py
# e.g. "Safety" → reads rag_crime_context / rag_crime_source from Neighborhood node
NEO4J_RAG_PROP: dict[str, str] = {
    "Safety":       "crime",
    "Housing":      "housing",
    "Grocery":      "grocery",
    "Healthcare":   "healthcare",
    "MBTA":         "transit",
    "Restaurants":  "restaurants",
    "Schools":      "schools",
    "Universities": "universities",
    "Bluebikes":    "bluebikes",
}

NEO4J_DOMAINS = [
    "Safety", "Housing", "Grocery", "Healthcare",
    "MBTA", "Restaurants", "Schools", "Universities", "Bluebikes",
]

NEO4J_TO_RAG_DOMAIN = {
    "Safety": "CRIME", "Housing": "HOUSING", "Grocery": "GROCERY",
    "Healthcare": "HEALTHCARE", "MBTA": "TRANSIT", "Restaurants": "RESTAURANTS",
    "Schools": "SCHOOLS", "Universities": "UNIVERSITIES", "Bluebikes": "BLUEBIKES",
}

DOMAIN_KEYWORDS = {
    "Safety":       ["crime", "safe", "safety", "violence", "theft", "assault",
                     "police", "incident", "robbery", "shooting"],
    "Housing":      ["housing", "rent", "price", "afford", "sqft", "property",
                     "buy", "home", "apartment", "condo", "assessed", "value"],
    "Grocery":      ["grocery", "supermarket", "food store", "market",
                     "whole foods", "trader joe", "star market", "essential store"],
    "Healthcare":   ["hospital", "clinic", "doctor", "health", "medical",
                     "urgent care", "pharmacy", "healthcare", "facility"],
    "MBTA":         ["mbta", "transit", "bus", "subway", "train", "commute",
                     "green line", "red line", "orange line", "blue line",
                     "silver line", "stop", "station", "rapid transit"],
    "Restaurants":  ["restaurant", "dining", "eat", "food", "cafe",
                     "bar", "cuisine", "takeout", "delivery", "yelp"],
    "Schools":      ["school", "elementary", "middle school", "high school",
                     "public school", "charter", "k-12", "district"],
    "Universities": ["university", "college", "higher education", "campus",
                     "mit", "harvard", "northeastern", "bu ", "boston university",
                     "student", "degree", "research"],
    "Bluebikes":    ["bluebikes", "bike share", "bicycle", "bikeshare",
                     "cycling", "bike station", "bike dock"],
}

# ── Neighbourhood constants ───────────────────────────────────────────────────

GREATER_BOSTON = [
    "BOSTON", "CAMBRIDGE", "SOMERVILLE", "QUINCY", "BROOKLINE",
    "ARLINGTON", "WATERTOWN", "CHELSEA", "EVERETT", "REVERE",
    "MEDFORD", "MALDEN", "NEWTON", "BELMONT", "LEXINGTON",
    "SALEM", "BEVERLY", "PEABODY", "MILTON",
    "ALLSTON", "BACK BAY", "BAY VILLAGE", "BEACON HILL",
    "BRIGHTON", "CHARLESTOWN", "CHINATOWN", "DORCHESTER",
    "DOWNTOWN", "EAST BOSTON", "FENWAY", "HYDE PARK",
    "JAMAICA PLAIN", "MATTAPAN", "MISSION HILL", "NORTH END",
    "ROSLINDALE", "ROXBURY", "SOUTH BOSTON", "SOUTH END",
    "WEST ROXBURY", "WEST END",
    "AREA 2/MIT", "CAMBRIDGEPORT", "EAST CAMBRIDGE",
    "MID CAMBRIDGE", "NORTH CAMBRIDGE", "WEST CAMBRIDGE",
    "WELLINGTON-HARRINGTON",
]

# Generic city names that only match when a domain qualifier word is also present
_GENERIC_NAMES = {"BOSTON", "CAMBRIDGE", "DOWNTOWN"}

_DOMAIN_QUALIFIERS = [
    "neighborhood", "area", "district", "section",
    "safety", "crime", "housing", "rent", "school", "restaurant",
    "transit", "grocery", "hospital", "score", "rank",
    "affordable", "safe", "walkable", "livability", "food",
    "commute", "bike", "weather", "university", "college",
]

# Phrases indicating a meta-question about the city — not a specific neighbourhood
_META_PHRASES = [
    "best neighborhood in", "worst neighborhood in",
    "neighborhoods in", "which neighborhood in",
    "neighborhood to live in", "neighborhoods to live",
]

# Nickname / compact → canonical
_MANUAL_ALIASES: dict[str, str] = {
    "SOUTHIE":        "SOUTH BOSTON",
    "SOUTHBOSTON":    "SOUTH BOSTON",
    "JP":             "JAMAICA PLAIN",
    "JAMAICAPLAIN":   "JAMAICA PLAIN",
    "EASTBOSTON":     "EAST BOSTON",
    "EASTIE":         "EAST BOSTON",
    "CHARLESTWON":    "CHARLESTOWN",
    "BEACONHILL":     "BEACON HILL",
    "BACKBAY":        "BACK BAY",
    "NORTHEND":       "NORTH END",
    "SOUTHEND":       "SOUTH END",
    "WESTEND":        "WEST END",
    "WESTROBURY":     "WEST ROXBURY",
    "WESTROXBURY":    "WEST ROXBURY",
    "HYDPARK":        "HYDE PARK",
    "HYDEPARK":       "HYDE PARK",
    "MISSIONHILL":    "MISSION HILL",
    "DOWNTOWNBOSTON": "DOWNTOWN",
    "FENWAYPARKWAY":  "FENWAY",
    "CAMBPORT":       "CAMBRIDGEPORT",
    "EASTCAMBRIDGE":  "EAST CAMBRIDGE",
    "NORTHCAMBRIDGE": "NORTH CAMBRIDGE",
    "WESTCAMBRIDGE":  "WEST CAMBRIDGE",
    "MIDCAMBRIDGE":   "MID CAMBRIDGE",
}

# If key canonical is matched, suppress the value canonicals (they are redundant)
_SUBSUMES: dict[str, set[str]] = {
    "DOWNTOWN":        {"BOSTON"},
    "EAST BOSTON":     {"BOSTON"},
    "SOUTH BOSTON":    {"BOSTON"},
    "WEST ROXBURY":    {"ROXBURY"},
    "EAST CAMBRIDGE":  {"CAMBRIDGE"},
    "NORTH CAMBRIDGE": {"CAMBRIDGE"},
    "WEST CAMBRIDGE":  {"CAMBRIDGE"},
    "MID CAMBRIDGE":   {"CAMBRIDGE"},
}


def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for hood in GREATER_BOSTON:
        alias_map[hood] = hood
        nospace = re.sub(r"\s+", "", hood)
        if nospace != hood:
            alias_map[nospace] = hood
    alias_map.update(_MANUAL_ALIASES)
    return alias_map


_ALIAS_MAP: dict[str, str] = _build_alias_map()


# ══════════════════════════════════════════════════════════════════════════════
# NEIGHBOURHOOD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _remove_subsumed(canonicals: list[str]) -> list[str]:
    """Remove canonicals made redundant by more-specific matches."""
    matched_set = set(canonicals)
    suppressed: set[str] = set()

    # Explicit pairs (DOWNTOWN suppresses BOSTON, etc.)
    for c in canonicals:
        for name in _SUBSUMES.get(c, set()):
            if name in matched_set:
                suppressed.add(name)

    # Word-level substring (ROXBURY ⊂ WESTROXBURY)
    nospace_map = {c: re.sub(r"\s+", "", c) for c in canonicals}
    nospace_set = set(nospace_map.values())
    for c, c_ns in nospace_map.items():
        if any(c_ns != o and c_ns in o for o in nospace_set):
            suppressed.add(c)

    return [c for c in canonicals if c not in suppressed]


def extract_all_neighborhoods(query: str, hint: Optional[str] = None) -> list[str]:
    """
    Return ALL neighbourhood canonical names found in the query, ordered by
    their position (first-mentioned first).

    Handles:
      - "Backbay" / "BackBay" / "back bay"  →  BACK BAY
      - "Southie"                            →  SOUTH BOSTON
      - "JP"                                 →  JAMAICA PLAIN
      - "Eastie"                             →  EAST BOSTON
      - "east boston restaurants"            →  [EAST BOSTON]  (BOSTON suppressed)
      - "west roxbury safety"                →  [WEST ROXBURY] (ROXBURY suppressed)
      - "Jamaica Plain vs Allston"           →  [JAMAICA PLAIN, ALLSTON]
      - "best neighborhood in Boston"        →  []  (meta-question)
    """
    if hint:
        return [hint.strip().upper()]

    q_lower   = query.lower()
    if any(p in q_lower for p in _META_PHRASES):
        return []

    q_clean   = re.sub(r"[^a-zA-Z0-9\s]", " ", query).upper()
    q_clean   = re.sub(r"\s+", " ", q_clean).strip()
    q_compact = re.sub(r"\s+", "", q_clean)

    found: dict[str, int] = {}  # canonical → earliest position

    for variant, canonical in _ALIAS_MAP.items():
        pos = q_clean.find(variant) if " " in variant else q_compact.find(variant)
        if pos == -1:
            continue
        if canonical in _GENERIC_NAMES:
            if not any(q in q_lower for q in _DOMAIN_QUALIFIERS):
                continue
        if canonical not in found or pos < found[canonical]:
            found[canonical] = pos

    ordered = [c for c, _ in sorted(found.items(), key=lambda x: x[1])]
    return _remove_subsumed(ordered)


def extract_neighborhood(query: str, hint: Optional[str] = None) -> Optional[str]:
    """Primary neighbourhood (first mentioned). None for meta-questions."""
    results = extract_all_neighborhoods(query, hint)
    return results[0] if results else None


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class GraphAgentState(TypedDict):
    query:             str
    neighborhood:      Optional[str]
    all_neighborhoods: list[str]
    domains:           list[str]

    graph_ctx_parts:   Annotated[list[dict], operator.add]
    struct_ctx_parts:  Annotated[list[dict], operator.add]
    rag_chunk_parts:   Annotated[list[list], operator.add]

    graph_ctx:  dict
    struct_ctx: dict
    rag_chunks: list[dict]

    draft:        str
    answer:       str
    val_verdict:  dict
    val_checks:   dict
    regenerated:  bool
    val_passed:   Optional[bool]
    val_attempts: int


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_domains(query: str) -> list[str]:
    q = query.lower()
    found = [d for d, kws in DOMAIN_KEYWORDS.items() if any(k in q for k in kws)]
    return found if found else NEO4J_DOMAINS[:]



def neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        # Suppress schema warnings (BORDERS, SERVED_BY) that are expected
        # when those relationships haven't been loaded yet
        notifications_min_severity="WARNING",
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )


# sf_connect removed — all retrieval now goes through Neo4j only.
# Snowflake mart queries are handled by the dedicated data_query / Cortex agent.


def neo4j_neighborhood_profile(driver, neighborhood: str) -> dict:
    with driver.session() as session:
        scores = [dict(r) for r in session.run("""
            MATCH (n:Neighborhood {name: $name})-[r:HAS_SCORE]->(d:Domain)
            RETURN d.name AS domain, r.composite_score AS score, r.grade AS grade,
                   r.avg_price_per_sqft AS price_sqft, r.avg_living_area_sqft AS living_area,
                   r.total_incidents AS total_incidents, r.violent_crime_count AS violent_crime,
                   r.total_stops AS transit_stops, r.has_rapid_transit AS has_rapid_transit,
                   r.total_restaurants AS restaurants, r.avg_rating AS restaurant_rating,
                   r.total_schools AS schools, r.total_universities AS universities,
                   r.total_stations AS bike_stations
            ORDER BY r.composite_score DESC
        """, name=neighborhood)]
        borders = [r["neighbor"] for r in session.run("""
            MATCH (n:Neighborhood {name: $name})-[:BORDERS]->(b:Neighborhood)
            RETURN b.name AS neighbor
        """, name=neighborhood)]
        mbta = [r["line"] for r in session.run("""
            MATCH (n:Neighborhood {name: $name})-[:SERVED_BY]->(m:MBTALine)
            RETURN m.name AS line
        """, name=neighborhood)]
        similar = [dict(r) for r in session.run("""
            MATCH (n:Neighborhood {name: $name})-[s:SIMILAR_TO]-(b:Neighborhood)
            RETURN b.name AS neighbor, s.avg_score_delta AS delta,
                   s.shared_domains AS shared_domains, s.based_on AS based_on
            ORDER BY s.avg_score_delta ASC LIMIT 5
        """, name=neighborhood)]
    return {"neighborhood": neighborhood, "domain_scores": scores,
            "borders": borders, "mbta_lines": mbta, "similar_to": similar}


def neo4j_top_by_domain(driver, domain: str, limit: int = 5) -> list[dict]:
    with driver.session() as session:
        return [dict(r) for r in session.run("""
            MATCH (n:Neighborhood)-[r:HAS_SCORE]->(d:Domain {name: $domain})
            RETURN n.name AS neighborhood, r.composite_score AS score, r.grade AS grade
            ORDER BY r.composite_score DESC LIMIT $limit
        """, domain=domain, limit=limit)]


def neo4j_neighborhood_rank(driver, neighborhood: str, domain: str) -> dict:
    with driver.session() as session:
        rows = [dict(r) for r in session.run("""
            MATCH (n:Neighborhood)-[r:HAS_SCORE]->(d:Domain {name: $domain})
            WHERE r.composite_score IS NOT NULL
            RETURN n.name AS neighborhood, r.composite_score AS score, r.grade AS grade
            ORDER BY r.composite_score DESC
        """, domain=domain)]
    total = len(rows)
    idx   = next((i for i, r in enumerate(rows) if r["neighborhood"] == neighborhood), None)
    if idx is None:
        return {"rank": None, "total": total,
                "summary": f"Rank not found for {neighborhood} in {domain}",
                "above": [], "below": []}
    rank  = idx + 1
    # Return ALL neighborhoods so synthesize_node can inject the complete
    # ranking — eliminating any reason for Claude to invent peer scores.
    return {
        "rank":    rank,
        "total":   total,
        "summary": (
            f"{neighborhood} ranks {rank} out of {total} neighborhoods "
            f"in Greater Boston for {domain} (score {rows[idx]['score']:.1f}, "
            f"grade {rows[idx]['grade']})"
        ),
        "above": rows[:idx],          # every neighborhood scoring higher
        "below": rows[idx + 1:],      # every neighborhood scoring lower
        "all_ranked": rows,           # full ordered list for complete reference
    }


def neo4j_transit_connected(driver, neighborhood: str) -> list[dict]:
    with driver.session() as session:
        return [dict(r) for r in session.run("""
            MATCH (n:Neighborhood {name: $name})-[:SERVED_BY]->(m:MBTALine)
                  <-[:SERVED_BY]-(b:Neighborhood)
            WHERE b.name <> $name
            RETURN b.name AS neighbor, m.name AS shared_line
            ORDER BY m.name, b.name
        """, name=neighborhood)]


# sf_housing_detail and sf_safety_detail removed.
# Structured mart queries are handled by the dedicated Cortex/data_query agent.
# This graph agent only queries Neo4j and RAG.


# Minimum keyword hits required for a chunk to be considered relevant.
# A chunk must contain at least this many domain signal words to be included.
_RAG_MIN_KEYWORD_HITS = 2

# Domain signal words used to assess chunk relevance at query time.
# If a stored chunk doesn't contain enough of these words it is skipped.
_RAG_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "Safety":       ["crime", "safety", "incident", "violent", "police", "theft",
                     "assault", "robbery", "shooting", "offense"],
    "Housing":      ["housing", "rent", "apartment", "price", "sqft", "affordable",
                     "mortgage", "condo", "property", "bedroom", "assessed"],
    "Grocery":      ["grocery", "supermarket", "market", "food store", "provisions"],
    "Healthcare":   ["hospital", "clinic", "health", "medical", "doctor",
                     "pharmacy", "urgent care"],
    "MBTA":         ["mbta", "transit", "subway", "bus", "train", "station",
                     "commute", "green line", "red line", "orange line"],
    "Restaurants":  ["restaurant", "dining", "food", "cafe", "cuisine",
                     "yelp", "takeout"],
    "Schools":      ["school", "education", "student", "district", "k-12",
                     "elementary", "high school"],
    "Universities": ["university", "college", "campus", "degree",
                     "mit", "harvard", "northeastern", "boston university"],
    "Bluebikes":    ["bluebike", "bike", "bicycle", "cycling", "bikeshare"],
}


def _chunk_is_relevant(text: str, domain: str) -> bool:
    """
    Return True if the chunk text contains enough domain signal words
    to be genuinely useful for the queried domain.

    Prevents injecting off-topic stored chunks into the prompt — e.g.
    a surveillance infrastructure report stored under 'crime' isn't
    useful for a question about whether a neighborhood is safe to live in.
    """
    if not text:
        return False
    signals = _RAG_DOMAIN_SIGNALS.get(domain, [])
    if not signals:
        return True  # no signals defined — include by default
    lower = text.lower()
    hits  = sum(1 for s in signals if s in lower)
    return hits >= _RAG_MIN_KEYWORD_HITS


def neo4j_rag_context(driver, neighborhood: str, domains: list[str]) -> list[dict]:
    """
    Read RAG context properties from the Neighborhood node in Neo4j,
    filtered to only the queried domains and only if the stored chunk
    is actually relevant to that domain.

    The neo4j_schema_loader stored the top chunk per domain as node properties:
      rag_crime_context / rag_crime_source
      rag_housing_context / rag_housing_source  ... etc.

    Relevance check: a chunk must contain at least _RAG_MIN_KEYWORD_HITS
    domain signal words — if the stored chunk is off-topic (e.g. a generic
    city report stored under a domain), it is skipped rather than injected.
    """
    props_to_fetch = []
    for domain in domains:
        prop_key = NEO4J_RAG_PROP.get(domain)
        if prop_key:
            props_to_fetch.append((domain, f"rag_{prop_key}_context", f"rag_{prop_key}_source"))

    if not props_to_fetch:
        return []

    return_clauses = ", ".join(
        f"n.`{ctx_prop}` AS {domain.lower()}_ctx, n.`{src_prop}` AS {domain.lower()}_src"
        for domain, ctx_prop, src_prop in props_to_fetch
    )

    with driver.session() as session:
        result = session.run(
            f"MATCH (n:Neighborhood {{name: $name}}) RETURN {return_clauses}",
            name=neighborhood,
        )
        row = result.single()

    if not row:
        return []

    chunks = []
    for domain, ctx_prop, src_prop in props_to_fetch:
        text = row.get(f"{domain.lower()}_ctx")
        src  = row.get(f"{domain.lower()}_src")
        if not text:
            log.debug(f"[rag] {domain}: no stored chunk on node")
            continue
        if not _chunk_is_relevant(text, domain):
            log.info(f"[rag] {domain}: chunk skipped — insufficient domain signal words")
            continue
        chunks.append({
            "domain":       domain,
            "DOMAIN":       domain.upper(),
            "chunk_text":   text,
            "CHUNK_TEXT":   text,
            "source_file":  src or "",
            "SOURCE_FILE":  src or "",
            "hybrid_score": 1.0,
            "similarity":   1.0,
        })
    return chunks


def _fmt_check(issues: list, fatal: bool = False, warn: bool = False) -> dict:
    if not issues:
        return {"status": "✅ PASS", "issues": []}
    if fatal:
        return {"status": "❌ FAIL", "issues": issues}
    if warn:
        return {"status": "⚠️  WARN", "issues": issues}
    return {"status": "❌ FAIL", "issues": issues}


def _extract_authoritative_scores(graph_ctx: dict) -> dict:
    """
    Extract verified domain scores for the queried neighborhood from
    neighborhood_ranks["all_ranked"] — the same rows used to build VERIFIED
    RANKING DATA, sourced directly from HAS_SCORE edge composite_score in Neo4j.

    Returns: {"Safety": {"score": 49.3, "grade": "MODERATE"}, ...}
    """
    scores = {}
    neighborhood = (graph_ctx.get("profile") or {}).get("neighborhood", "")

    for domain, rank_data in (graph_ctx.get("neighborhood_ranks") or {}).items():
        # Find this neighborhood's row in the full ranked list
        all_ranked = rank_data.get("all_ranked", [])
        match = next(
            (r for r in all_ranked if r.get("neighborhood", "").upper() == neighborhood.upper()),
            None
        )
        if match:
            scores[domain] = {
                "score": float(match["score"]),
                "grade": match.get("grade", ""),
            }
            continue

        # Fallback: parse summary string
        summary = rank_data.get("summary", "")
        if "(score " in summary and ", grade " in summary:
            try:
                score_part = summary.split("(score ")[1].split(",")[0].strip()
                grade_part = summary.split(", grade ")[1].split(")")[0].strip()
                scores[domain] = {"score": float(score_part), "grade": grade_part}
            except (IndexError, ValueError):
                pass

    return scores


SYSTEM_PROMPT = """You are the NeighbourWise AI graph agent for Greater Boston neighborhood
livability analysis.

══════════════════════════════════════════════════════
ANTI-HALLUCINATION RULES — READ BEFORE WRITING ANYTHING
══════════════════════════════════════════════════════
1. PEER COMPARISONS: The user message contains "=== VERIFIED RANKING DATA ===".
   This is the ONLY source you may use for citing other neighborhoods' scores.
   Every neighborhood name + score + grade you mention MUST appear verbatim there.

2. NO TRAINING DATA: Do not use your training knowledge for Boston neighborhood
   scores. Those figures may be outdated or estimated. The VERIFIED RANKING DATA
   is the sole ground truth for comparisons.

3. IF A SCORE IS NOT IN VERIFIED RANKING DATA: Do not cite it. Write instead:
   "Peer comparison data not available" or omit the comparison entirely.

4. SCOPE: Answer only the domains the query explicitly asks about.
   "Is Allston safe and affordable?" = Safety + Housing only.
   Do NOT add MBTA, Restaurants, Grocery, etc. unless the query requests them.

5. SCORE AUTHORITY: VERIFIED RANKING DATA is the single source of truth.
   Any score you write that does not appear verbatim in the ranking tables
   below is a hallucination — even if you are confident in the number.

6. NEIGHBOR NAMES: If you see neighborhood names in the Graph Profile JSON
   (in similar_to, borders, or transit data), you MUST NOT look up or recall
   their scores. Only cite scores that appear in the VERIFIED RANKING DATA.
══════════════════════════════════════════════════════

Response format:
  - Lead with a direct answer to the user's question
  - State the queried neighborhood's score, grade, and rank from VERIFIED RANKING DATA
  - REQUIRED: cite the key detail metrics from Graph Profile domain_metrics:
      Safety  → total_incidents, violent_crime_count
      Housing → avg_price_per_sqft, avg_living_area_sqft
      MBTA    → total_stops, has_rapid_transit
      (include whichever domains the query is about)
  - For peer comparisons: ONLY cite neighborhoods whose scores appear in the ranking tables
  - Note INSUFFICIENT DATA if a domain has score 0
  - Keep response between 300-500 words
  - End with: "Sources: [graph] [RAG chunks]"
Never fabricate scores or relationships not present in the provided context."""


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def plan_node(state: GraphAgentState) -> dict:
    """Detect domains + all neighbourhood names, resolve primary neighbourhood."""
    query = state["query"]
    hint  = state.get("neighborhood")

    domains           = detect_domains(query)
    all_neighborhoods = extract_all_neighborhoods(query, hint)
    neighborhood      = all_neighborhoods[0] if all_neighborhoods else None

    log.info(
        f"[plan] domains={domains}  "
        f"neighborhood={neighborhood}  "
        f"all_neighborhoods={all_neighborhoods}"
    )

    return {
        "domains":           domains,
        "neighborhood":      neighborhood,
        "all_neighborhoods": all_neighborhoods,
        "graph_ctx_parts":   [],
        "struct_ctx_parts":  [],
        "rag_chunk_parts":   [],
    }


def neo4j_node(state: GraphAgentState) -> dict:
    """Parallel retrieval — Neo4j graph traversal."""
    t0           = time.time()
    neighborhood = state.get("neighborhood")
    domains      = state.get("domains", [])
    ctx          = {}

    try:
        driver = neo4j_driver()
        if neighborhood:
            ctx["profile"]         = neo4j_neighborhood_profile(driver, neighborhood)
            ctx["transit_network"] = neo4j_transit_connected(driver, neighborhood)
        ctx["top_by_domain"] = {
            d: neo4j_top_by_domain(driver, d, limit=5)
            for d in domains
        }
        if neighborhood:
            ctx["neighborhood_ranks"] = {
                d: neo4j_neighborhood_rank(driver, neighborhood, d)
                for d in domains
            }
        driver.close()
        log.info(f"[neo4j_node] complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        log.warning(f"[neo4j_node] failed: {e}")
        ctx = {"error": str(e)}

    return {"graph_ctx_parts": [ctx]}


def mart_node(state: GraphAgentState) -> dict:
    """
    No-op — structured mart queries are handled by the Cortex/data_query agent.
    This graph agent is Neo4j + RAG only. Returning empty struct_ctx.
    """
    return {"struct_ctx_parts": [{}]}


def rag_node(state: GraphAgentState) -> dict:
    """
    RAG retrieval — reads rag_*_context properties from the Neighborhood node
    in Neo4j.  No Snowflake call needed: neo4j_schema_loader already stored
    the top chunk per domain directly on each node during graph build.

    For general queries with no specific neighborhood, returns empty chunks
    (the graph + ranking data is sufficient context in that case).
    """
    t0           = time.time()
    neighborhood = state.get("neighborhood")
    domains      = state.get("domains", [])
    chunks: list[dict] = []

    if not neighborhood:
        log.info("[rag_node] no neighborhood — skipping RAG context retrieval")
        return {"rag_chunk_parts": [[]]}

    try:
        driver = neo4j_driver()
        chunks = neo4j_rag_context(driver, neighborhood, domains)
        driver.close()
        log.info(
            f"[rag_node] complete ({time.time()-t0:.1f}s) — "
            f"{len(chunks)} domain chunks from Neo4j node properties"
        )
    except Exception as e:
        log.warning(f"[rag_node] failed: {e}")

    return {"rag_chunk_parts": [chunks]}


def merge_node(state: GraphAgentState) -> dict:
    graph_ctx  = state["graph_ctx_parts"][0]  if state.get("graph_ctx_parts")  else {}
    struct_ctx = state["struct_ctx_parts"][0] if state.get("struct_ctx_parts") else {}
    rag_chunks = state["rag_chunk_parts"][0]  if state.get("rag_chunk_parts")  else []

    log.info(
        f"[merge_node] graph_ctx keys={list(graph_ctx.keys())} | "
        f"struct_ctx keys={list(struct_ctx.keys())} | "
        f"rag_chunks={len(rag_chunks)}"
    )
    return {"graph_ctx": graph_ctx, "struct_ctx": struct_ctx, "rag_chunks": rag_chunks}


def synthesize_node(state: GraphAgentState) -> dict:
    t0         = time.time()
    query      = state["query"]
    graph_ctx  = state.get("graph_ctx", {})
    struct_ctx = state.get("struct_ctx", {})
    rag_chunks = state.get("rag_chunks", [])

    parts = []

    # Pre-formatted ranking sentences (anti-hallucination: inject as plain text
    # before raw JSON so Claude reads verified facts first)
    rank_sentences = []
    for domain, rank_data in (graph_ctx.get("neighborhood_ranks") or {}).items():
        if rank_data.get("summary"):
            rank_sentences.append(f"\n--- {domain.upper()} RANKING ---")
            rank_sentences.append(f"  {rank_data['summary']}")

        # Inject the complete ordered list for this domain.
        # Every neighborhood appears with its verified score and grade.
        # Claude MUST only cite numbers from this list — no training-memory scores.
        all_ranked = rank_data.get("all_ranked", [])
        if all_ranked:
            rank_sentences.append(f"  COMPLETE {domain.upper()} RANKING (all {len(all_ranked)} neighborhoods):")
            for i, r in enumerate(all_ranked, 1):
                marker = " ◀ QUERIED" if r["neighborhood"] == (
                    graph_ctx.get("profile", {}).get("neighborhood", "")
                ) else ""
                rank_sentences.append(
                    f"    {i:2}. {r['neighborhood'].title():<30} score {r['score']:.1f}  {r['grade']}{marker}"
                )

    if rank_sentences:
        parts.append(
            "=== VERIFIED RANKING DATA ===\n"
            "ABSOLUTE RULE: Every neighborhood name + score you cite MUST appear\n"
            "verbatim in the complete ranking lists below. DO NOT use any score\n"
            "from training memory — those figures are unreliable and prohibited.\n"
            + "\n".join(rank_sentences)
        )

    if graph_ctx:
        graph_stripped = {
            k: v for k, v in graph_ctx.items()
            if k not in ("top_by_domain", "neighborhood_ranks")
        }
        # Clean the profile before injection:
        # - similar_to / borders: list neighbor names → Claude recalls their scores
        # - transit_network: same risk
        # - domain_scores score/grade: already in VERIFIED RANKING DATA — keep only
        #   the detail metrics (incidents, sqft, rent etc.) which are NOT in rankings
        _SCORE_GRADE_KEYS = {"score", "grade"}
        if "profile" in graph_stripped and isinstance(graph_stripped["profile"], dict):
            prof = graph_stripped["profile"]
            # Rebuild domain_scores keeping only detail metrics, not score/grade
            domain_metrics = []
            for ds in prof.get("domain_scores") or []:
                metrics = {k: v for k, v in ds.items() if k not in _SCORE_GRADE_KEYS}
                if len(metrics) > 1:   # more than just "domain" key → has real metrics
                    domain_metrics.append(metrics)
            profile_clean = {
                k: v for k, v in prof.items()
                if k not in {"similar_to", "borders"}
            }
            if domain_metrics:
                profile_clean["domain_metrics"] = domain_metrics
            profile_clean.pop("domain_scores", None)
            graph_stripped = {**graph_stripped, "profile": profile_clean}
        graph_stripped.pop("transit_network", None)
        parts.append("\n=== GRAPH PROFILE (Neo4j) ===")
        parts.append(json.dumps(graph_stripped, indent=2, default=str))

    # struct_ctx intentionally not injected — mart queries handled by data_query agent

    if rag_chunks:
        parts.append("\n=== UNSTRUCTURED CONTEXT (RAG Chunks) ===")
        for i, c in enumerate(rag_chunks, 1):
            parts.append(
                f"[{i}] Domain: {c.get('domain','?')} | "
                f"Source: {c.get('source_file','?')} | "
                f"Score: {c.get('hybrid_score', 0):.3f}\n"
                f"{c.get('chunk_text','')[:600]}"
            )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp   = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=700, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"User query: {query}\n\n" + "\n".join(parts)}],
    )
    draft = resp.content[0].text
    log.info(f"[synthesize_node] complete ({time.time()-t0:.1f}s)  {len(draft)} chars")
    return {"draft": draft}


def validate_node(state: GraphAgentState) -> dict:
    t0         = time.time()
    query      = state["query"]
    draft      = state["draft"]
    graph_ctx  = state.get("graph_ctx", {})
    struct_ctx = state.get("struct_ctx", {})
    rag_chunks = state.get("rag_chunks", [])

    val_verdict  = {}
    val_checks   = {}
    regenerated  = False
    val_passed   = None
    val_attempts = 1
    answer       = draft

    try:
        # Extract authoritative scores from Neo4j — injected into graph_ctx
        # so the universal validator can verify any peer score Claude cites.
        authoritative_scores = _extract_authoritative_scores(graph_ctx)
        if authoritative_scores:
            log.info(f"[validate_node] authoritative scores from Neo4j: {authoritative_scores}")

        # Build compact verified-scores string (one line per neighborhood per domain).
        # UniversalValidator._build_graph_validation_context uses [:4000] for graph_ctx
        # so all 51 neighborhoods survive for up to ~4 queried domains.
        # Queried neighborhood sorted first so it is never truncated.
        neighborhood_name = (graph_ctx.get("profile") or {}).get("neighborhood", "")
        verified_lines = []
        for domain, rank_data in (graph_ctx.get("neighborhood_ranks") or {}).items():
            for row in rank_data.get("all_ranked") or []:
                line = (f"{row['neighborhood'].title()} {domain} "
                        f"{row['score']:.1f} {row.get('grade','')}")
                if row["neighborhood"].upper() == neighborhood_name.upper():
                    verified_lines.insert(0, line)
                else:
                    verified_lines.append(line)

        # Build domain_metrics from profile.domain_scores — strip score/grade,
        # keep detail figures (incidents, sqft, rent etc.).
        # Note: domain_metrics only exists as a local variable in synthesize_node;
        # the raw graph_ctx.profile still has domain_scores, not domain_metrics.
        _SG = {"score", "grade"}
        domain_metrics = [
            {k: v for k, v in ds.items() if k not in _SG}
            for ds in (graph_ctx.get("profile") or {}).get("domain_scores") or []
            if sum(1 for k in ds if k not in _SG) > 1   # has real metrics beyond domain name
        ]

        # ── v4.3 FIX: Pre-format domain_metrics AND rank positions ─────
        # _build_graph_validation_context() expects a pre-formatted STRING
        # under the key "verified_detail_metrics" — NOT a raw list of dicts
        # under "domain_metrics".  Without this, GPT-4o never sees the detail
        # figures (total_incidents, avg_price_per_sqft, etc.) labeled with the
        # queried neighborhood name, and flags correctly-cited numbers as
        # fabricated data.
        #
        # v4.3 addition: also inject rank summaries (e.g. "ranks 29 out of 51")
        # so GPT-4o can verify rank position claims in the draft.  Previously
        # these were only in the synthesize prompt but absent from the validator
        # context, causing false fabricated_data flags on correct rank citations.
        detail_lines = []

        # Inject rank positions first — these are verified from neo4j_neighborhood_rank()
        for domain, rank_data in (graph_ctx.get("neighborhood_ranks") or {}).items():
            summary = rank_data.get("summary", "")
            if summary:
                detail_lines.append(f"{neighborhood_name} {domain} RANK: {summary}")

        # Then inject detail metrics (incidents, sqft, rent, etc.)
        for dm in domain_metrics:
            domain_label = dm.get("domain", "Unknown")
            metric_parts = [
                f"{k}={v}" for k, v in dm.items()
                if k != "domain" and v is not None
            ]
            if metric_parts:
                detail_lines.append(
                    f"{neighborhood_name} {domain_label}: {', '.join(metric_parts)}"
                )

        # Pass a focused graph_ctx to the validator:
        #   authoritative_scores     — queried neighborhood's verified scores
        #   verified_peer_scores     — all 51 hoods per domain as flat string
        #   verified_detail_metrics  — detail figures labeled with neighborhood name
        graph_ctx_for_validator = {
            "authoritative_scores":    authoritative_scores,
            "verified_peer_scores":    "\n".join(verified_lines),
            "verified_detail_metrics": "\n".join(detail_lines),
        }

        # Call UniversalValidator directly (not the convenience wrapper) so we
        # retain the full CheckResult including details (score, raw_issues, etc.)
        validator = UniversalValidator(conn=None)
        val_result = validator.validate(AgentType.GRAPH_QUERY, {
            "query":        query,
            "answer":       draft,
            "graph_ctx":    graph_ctx_for_validator,
            "struct_ctx":   {},
            "rag_chunks":   rag_chunks,
            "neighborhood": neighborhood_name,
            "domains":      state.get("domains", []),
        })

        # Extract results — val_result.result holds improved answer (or original)
        answer      = val_result.result if val_result.result else draft
        regenerated = val_result.improved
        val_passed  = val_result.passed

        # Pull score and raw_issues from the GPT-4o check's details dict
        raw_issues   = {}
        score        = 95 if val_passed else 60
        val_attempts = 1
        for check_name, check in val_result.checks.items():
            if check_name == "gpt4o_graph_validation":
                raw_issues = check.details.get("raw_issues", {})
                score      = check.details.get("score", score)
                break

        val_verdict = {
            "score":  score,
            "issues": raw_issues,
            "passed": val_passed,
        }
        val_checks = {
            "score_errors":      _fmt_check(raw_issues.get("score_errors", [])),
            "grade_errors":      _fmt_check(raw_issues.get("grade_errors", [])),
            "fabricated_data":   _fmt_check(raw_issues.get("fabricated_data", []), fatal=True),
            "missing_insights":  _fmt_check(raw_issues.get("missing_insights", []), warn=True),
            "comparison_errors": _fmt_check(raw_issues.get("comparison_errors", [])),
            "richness_issues":   _fmt_check(raw_issues.get("richness_issues", []), warn=True),
        }
        log.info(
            f"[validate_node] complete ({time.time()-t0:.1f}s) — "
            f"passed={val_passed}  score={score}/100  regenerated={regenerated}"
        )
    except Exception as e:
        log.warning(f"[validate_node] failed (non-fatal): {e}")

    return {
        "answer": answer, "val_verdict": val_verdict, "val_checks": val_checks,
        "regenerated": regenerated, "val_passed": val_passed, "val_attempts": val_attempts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING + GRAPH BUILD
# ══════════════════════════════════════════════════════════════════════════════

def dispatch_retrieval(state: GraphAgentState) -> list[Send]:
    return [
        Send("neo4j_node", state),
        Send("mart_node",  state),
        Send("rag_node",   state),
    ]


def _build_graph() -> StateGraph:
    builder = StateGraph(GraphAgentState)
    builder.add_node("plan_node",       plan_node)
    builder.add_node("neo4j_node",      neo4j_node)
    builder.add_node("mart_node",       mart_node)
    builder.add_node("rag_node",        rag_node)
    builder.add_node("merge_node",      merge_node)
    builder.add_node("synthesize_node", synthesize_node)
    builder.add_node("validate_node",   validate_node)

    builder.set_entry_point("plan_node")
    builder.add_conditional_edges(
        "plan_node", dispatch_retrieval,
        ["neo4j_node", "mart_node", "rag_node"],
    )
    builder.add_edge("neo4j_node", "merge_node")
    builder.add_edge("mart_node",  "merge_node")
    builder.add_edge("rag_node",   "merge_node")
    builder.add_edge("merge_node",      "synthesize_node")
    builder.add_edge("synthesize_node", "validate_node")
    builder.add_edge("validate_node",   END)

    return builder.compile()


_GRAPH = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def ask_graph_agent(query: str, neighborhood: str = None) -> dict:
    """
    Full graph agent pipeline (LangGraph parallel retrieval).

    v4.3: fixed validator rank position verification — rank summaries now
    included in verified_detail_metrics so GPT-4o can verify "ranks X out of Y".

    v4.2: fixed validator domain_metrics key mismatch — detail figures now
    reach GPT-4o as labeled sentences under "verified_detail_metrics".

    v4.1: neighbourhood extraction completely rewritten.
    "Backbay", "Southie", "JP", "Eastie", and all 51 Greater Boston names
    (including compound variants like "west roxbury" / "westroxbury") resolve
    correctly to their canonical form before any DB call is made.
    """
    t0 = time.time()
    log.info(f"[ask_graph_agent] query={query!r}  neighborhood={neighborhood!r}")

    initial_state: GraphAgentState = {
        "query":             query,
        "neighborhood":      neighborhood,
        "all_neighborhoods": [],
        "domains":           [],
        "graph_ctx_parts":   [],
        "struct_ctx_parts":  [],
        "rag_chunk_parts":   [],
        "graph_ctx":         {},
        "struct_ctx":        {},
        "rag_chunks":        [],
        "draft":             "",
        "answer":            "",
        "val_verdict":       {},
        "val_checks":        {},
        "regenerated":       False,
        "val_passed":        None,
        "val_attempts":      1,
    }

    final_state  = _GRAPH.invoke(initial_state)
    graph_ctx    = final_state.get("graph_ctx",   {})
    struct_ctx   = final_state.get("struct_ctx",  {})
    rag_chunks   = final_state.get("rag_chunks",  [])
    val_verdict  = final_state.get("val_verdict", {})
    val_checks   = final_state.get("val_checks",  {})
    regenerated  = final_state.get("regenerated", False)
    val_passed   = final_state.get("val_passed",  None)
    val_attempts = final_state.get("val_attempts", 1)
    answer       = final_state.get("answer", "")

    all_issues = [i for issues in (val_verdict.get("issues") or {}).values()
                  for i in (issues or [])]
    any_fatal  = bool((val_verdict.get("issues") or {}).get("fabricated_data"))

    log.info(f"[ask_graph_agent] total={time.time()-t0:.1f}s")

    return {
        "type":              "graph_query",
        "answer":            answer,
        "sql":               None,
        "results":           [],
        "rag_chunks":        rag_chunks,
        "improved":          regenerated,
        "validation": {
            "checks":            val_checks,
            "needs_improvement": any_fatal or (not val_passed),
            "total_issues":      len(all_issues),
            "all_issues":        all_issues,
            "passed":            val_passed,
            "score":             val_verdict.get("score"),
            "regenerated":       regenerated,
            "attempts":          val_attempts,
        },
        "neighborhood":      final_state.get("neighborhood"),
        "all_neighborhoods": final_state.get("all_neighborhoods", []),
        "domains":           final_state.get("domains", []),
        "graph_data": {
            "profile":         graph_ctx.get("profile"),
            "top_by_domain":   graph_ctx.get("top_by_domain", {}),
            "transit_network": graph_ctx.get("transit_network", []),
        },
        "sources": {
            "graph_nodes": bool(graph_ctx and "error" not in graph_ctx),
            "rag_chunks":  len(rag_chunks),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display_result(result: dict):
    SEP  = "─" * 65
    SEP2 = "═" * 65

    if result.get("error"):
        print(f"\n❌  {result['error']}\n")
        return

    all_hoods = result.get("all_neighborhoods", [])
    hood_str  = " + ".join(all_hoods) if all_hoods else "No neighborhood detected"

    print(f"\n{SEP2}")
    print(f"  GRAPH AGENT  —  {hood_str}")
    print(f"{SEP2}\n")

    for line in result.get("answer", "").splitlines():
        print(textwrap.fill(line, width=78) if len(line) > 78 else line)
    print()

    s = result.get("sources", {})
    print(f"{SEP}")
    print(
        f"  Sources  →  "
        f"Graph: {'✓' if s.get('graph_nodes') else '✗'}  |  "
        f"RAG chunks: {s.get('rag_chunks', 0)}"
    )
    print(f"  Domains  →  {', '.join(result.get('domains', []))}")

    val = result.get("validation", {})
    if val:
        passed_str = "✅ Passed" if val.get("passed") else "🔍 Regenerated"
        regen_str  = f"  (attempt {val.get('attempts', 1)})" if val.get("regenerated") else ""
        print(f"{SEP}")
        print(f"  Validator  →  {passed_str}{regen_str}  |  Score: {val.get('score')}/100")
        for name, c in (val.get("checks") or {}).items():
            if c.get("issues"):
                print(f"     {c.get('status','')}  {name}")
                for issue in c["issues"]:
                    print(f"          → {issue[:110]}")

    rag = result.get("rag_chunks", [])
    if rag:
        print(f"{SEP}")
        print("  RAG sources:")
        for c in rag[:3]:
            domain = c.get("DOMAIN", c.get("domain", "?"))
            src    = Path(c.get("SOURCE_FILE", c.get("source_file", "?"))).name
            sim    = c.get("similarity", c.get("hybrid_score", 0))
            print(f"     [{domain}]  {src}  (score {sim:.3f})")

    print(f"{SEP2}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="NeighbourWise Graph Agent (LangGraph) — terminal interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python Graph_agent.py -q "Is Backbay safe and affordable?"
              python Graph_agent.py -q "Best transit access" -n CAMBRIDGE
              python Graph_agent.py -i
        """)
    )
    p.add_argument("-q", "--query",        help="Single query string")
    p.add_argument("-n", "--neighborhood", help="Neighborhood hint (e.g. ALLSTON)")
    p.add_argument("-i", "--interactive",  action="store_true")
    p.add_argument("--json",               action="store_true",
                   help="Print raw JSON instead of formatted output")
    return p, p.parse_args()


def run_query(query: str, neighborhood: str = None, as_json: bool = False):
    print(f"\n[Graph Agent] Query: {query!r}")
    if neighborhood:
        print(f"[Graph Agent] Neighborhood hint: {neighborhood}")
    print()
    try:
        result = ask_graph_agent(query, neighborhood=neighborhood)
    except Exception as e:
        print(f"\n❌  Pipeline error: {e}\n")
        return
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        display_result(result)


def main():
    print(f"  [env] Loaded .env from: {_env_path}")
    parser, args = parse_args()

    if not sys.stdin.isatty() and not args.query and not args.interactive:
        query = sys.stdin.read().strip()
        if query:
            run_query(query, neighborhood=args.neighborhood, as_json=args.json)
        return

    if args.query:
        run_query(args.query, neighborhood=args.neighborhood, as_json=args.json)
        return

    if args.interactive:
        print("\n" + "═" * 65)
        print("  NeighbourWise Graph Agent (LangGraph) — Interactive")
        print("  Type a question, or 'exit' to quit.")
        print("═" * 65)
        while True:
            try:
                query = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break
            run_query(query, neighborhood=args.neighborhood, as_json=args.json)
        return

    parser.print_help()


if __name__ == "__main__":
    main()