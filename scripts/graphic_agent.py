import os
import sys
import json
import re
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector
import altair as alt
import vl_convert as vlc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from openai import OpenAI
from tavily import TavilyClient

# ── Universal Validator (replaces validator_agent) ────────────────────────────
from universal_validator import UniversalValidator, AgentType
MAX_RETRIES = 3

# ── env ────────────────────────────────────────────────────────────────────────
load_dotenv()
SNOWFLAKE_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER      = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE  = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_ROLE      = os.getenv("SNOWFLAKE_ROLE")
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

OUTPUT_BASE = Path(__file__).parent / "outputs" / "graphic_agent"
CHARTS_DIR  = OUTPUT_BASE / "charts"
IMAGES_DIR  = OUTPUT_BASE / "images"
CACHE_FILE  = OUTPUT_BASE / "landmarks_cache.json"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ── Image Perspectives ─────────────────────────────────────────────────────────
IMAGE_PERSPECTIVES = [
    {
        "label"       : "landmark",
        "description" : "main landmark and commercial street",
        "prompt_focus": (
            "The main commercial street and landmark exterior of the neighborhood. "
            "Focus on the most iconic architectural feature visible from the street. "
            "{visual} "
            "Wide street view, modest storefronts with plain solid-color awnings "
            "at ground level, pedestrians, parked cars. "
            "Daytime, clear sky, sharp architectural detail."
        )
    },
    {
        "label"       : "residential",
        "description" : "residential side street character",
        "prompt_focus": (
            "A quiet residential side street one block off the main road. "
            "{visual} "
            "No commercial signage, no text visible anywhere — purely residential. "
            "Late afternoon light filtering through street trees."
        )
    },
    {
        "label"       : "transit",
        "description" : "transit and daily commuter life",
        "prompt_focus": (
            "Street-level scene focused on daily commuter life and transit. "
            "{transit} "
            "People waiting at a bus stop or subway entrance, carrying bags, "
            "checking phones, diverse mix of commuters. "
            "Rush hour feel, busy intersection, traffic lights, crosswalk markings. "
            "ZERO legible text anywhere in the entire scene — "
            "no words on transit shelters, no readable station names, "
            "no STOP signs, no street signs, no route numbers, "
            "no schedules, no posters, no any readable letters at all. "
            "All signage must be blank, blurred, or face away from camera. "
            "Authentic working neighborhood energy — not a tourist area."
        )
    },
    {
        "label"       : "food_nightlife",
        "description" : "restaurant and nightlife strip at dusk",
        "prompt_focus": (
            "The neighborhood restaurant strip at dusk. "
            "Ground floor restaurants and bars with plain illuminated awnings "
            "in solid colors, NO readable text or sign words visible anywhere, "
            "outdoor seating with people eating and drinking, warm light "
            "spilling from windows onto the brick sidewalk. "
            "2-3 story brick buildings above the restaurants with residential "
            "apartments, iron fire escapes on the facades. "
            "Dusk lighting — sky transitioning from orange to deep blue. "
            "Modest American urban restaurant strip — not a bazaar or market."
        )
    }
]

# ── Neighborhood Visual Overrides ──────────────────────────────────────────────
NEIGHBORHOOD_OVERRIDES = {
    "FENWAY": {
        "names": ["Fenway Park", "CITGO Sign", "Lansdowne Street"],
        "visual": (
            "A massive fully enclosed baseball stadium with a tall continuous "
            "exterior wall of dark forest-green painted concrete and steel rising "
            "4 to 5 stories above the street, completely solid with no openings, "
            "running the full length of the city block. The green painted wall "
            "has riveted steel panels, narrow pedestrian gates at street level, "
            "and old-style light stanchions mounted on top of the exterior wall. "
            "The sidewalk directly in front of the green wall is wide brick, "
            "with street vendors and souvenir stands. Across the narrow street "
            "are low 3-story brick bars and restaurants with plain green awnings "
            "and no readable text visible on any sign. In the far background "
            "above rooftops, a large triangular neon sign structure in red and "
            "blue glows on top of a building. The street is narrow, one lane "
            "each direction, parked cars and pedestrians on both sides. "
            "No interior of the stadium is visible anywhere."
        )
    },
    "ROXBURY": {
        "names": ["Dudley Square", "Nubian Square", "triple-deckers"],
        "visual": (
            "A working-class Boston neighborhood with a mix of two housing types "
            "side by side. On the main streets: modest 2-3 story attached brick "
            "rowhouses with flat or slightly pitched roofs, small front stoops "
            "of 3-4 steps with simple iron railings, ground floor units at "
            "near street level, faded brick in dark reds and browns, some with "
            "small bay windows. On the residential side streets: classic Boston "
            "wooden triple-decker houses 3 stories tall, each floor has a flat "
            "covered porch running the full width of the house, wood siding "
            "painted in faded yellows, greens and grays, small front yards "
            "with chain-link fences, narrow driveways between houses. "
            "Corner intersections have 2-3 story flat-roofed brick commercial "
            "buildings with plain solid-color awnings at street level, "
            "no readable text on any storefront. Large abstract community "
            "murals in bold colors on blank brick side walls. Wide concrete "
            "sidewalks, overhead utility wires on wooden poles, parked cars "
            "on both sides. Real modest working Boston neighborhood."
        )
    },
    "BEACON HILL": {
        "names": ["Massachusetts State House", "Acorn Street", "Boston Common"],
        "visual": (
            "Narrow cobblestone streets lined with uniform attached 4-story "
            "Federal-style red brick rowhouses sharing walls along the full "
            "length of the block. Each rowhouse has a white marble front stoop "
            "of 6-8 steps with black cast iron railings, black painted shutters "
            "on every window, black iron gas lanterns mounted on the brick "
            "facade beside the front door, window boxes with seasonal flowers. "
            "A gleaming gold dome of a government building visible above the "
            "rooftops at the top of the hill. Streets are very narrow cobblestone "
            "running steeply uphill, brick sidewalks uneven and narrow. "
            "No text visible anywhere."
        )
    },
    "BACK BAY": {
        "names": ["Prudential Tower", "Newbury Street", "Copley Square"],
        "visual": (
            "Wide tree-lined boulevard with a continuous row of attached "
            "4-5 story brownstone rowhouses running the full length of the "
            "block. Each brownstone has a distinctive bow bay window on every "
            "floor projecting out from the facade, a high front stoop of "
            "10-12 brownstone steps with ornate iron railings, arched "
            "doorways with transom windows, rusticated stone base. "
            "Ground floors of some rowhouses converted to boutique shops "
            "with plain solid-color awnings, outdoor cafe seating on wide "
            "brick sidewalks. A 52-story glass and steel skyscraper with "
            "stepped aluminum crown visible above the rooflines in the "
            "background. Central grassy boulevard median with mature "
            "American elm trees forming a full canopy over the sidewalk. "
            "No text visible on any storefront or sign."
        )
    },
    "SOUTH END": {
        "names": ["Tremont Street", "SoWa Art District", "Union Park"],
        "visual": (
            "Long continuous rows of attached 4-story Victorian brownstone "
            "and red brick rowhouses sharing walls along every block. "
            "Each rowhouse has an ornate bow bay window on every floor, "
            "a high front stoop of 8-10 brownstone steps with decorative "
            "cast iron railings, arched doorways with carved stone lintels "
            "and transom windows above. Ground floor units converted to art "
            "galleries and small restaurants with plain awnings, no readable "
            "text. Tree-lined streets with small fenced park squares."
        )
    },
    "NORTH END": {
        "names": ["Old North Church", "Hanover Street", "Paul Revere Mall"],
        "visual": (
            "Extremely dense 3-4 story attached red brick buildings packed "
            "tightly along very narrow streets barely wide enough for one car. "
            "Ground floors have Italian restaurant awnings in solid red, white "
            "and green with no readable text, small bakeries with pastry displays "
            "in windows. Upper floors residential with laundry hanging between "
            "buildings. A tall white colonial church steeple visible above "
            "the rooflines. Narrow uneven brick sidewalks, no trees."
        )
    },
    "CHINATOWN": {
        "names": ["Chinatown Gate", "Beach Street", "Tyler Street"],
        "visual": (
            "A large ornate ceremonial gateway with a green glazed tile "
            "pagoda roof, four red columns with gold detailing, spanning "
            "the entrance to a narrow street. Dense 4-6 story brick "
            "buildings with red and gold decorative facade elements, "
            "roast duck hanging in restaurant windows, produce crates on "
            "the sidewalk, red paper lanterns strung overhead. "
            "No readable text anywhere."
        )
    },
    "CHARLESTOWN": {
        "names": ["Bunker Hill Monument", "City Square", "Main Street"],
        "visual": (
            "Rows of attached 3-4 story Federal and Greek Revival red brick "
            "rowhouses along narrow one-way streets. Each rowhouse has a "
            "modest front stoop of 4-6 brick steps with simple iron railings, "
            "double-hung windows with black shutters, flat rooflines uniform "
            "across the block. Some blocks have wooden triple-deckers on side "
            "streets. A tall granite obelisk monument visible above the rooftops."
        )
    },
    "EAST BOSTON": {
        "names": ["Maverick Square", "Bremen Street Park", "Central Square"],
        "visual": (
            "Streets lined predominantly with wooden triple-decker houses "
            "3 stories tall, each floor with a flat covered porch, wood "
            "siding in pale yellows, blues, greens, small front yards with "
            "chain-link fences. Some brick rowhouses mixed in. Commercial "
            "streets have 2-story flat-roofed brick buildings with plain "
            "awnings, no readable text. Working-class immigrant character."
        )
    },
    "MISSION HILL": {
        "names": ["Brigham Circle", "Tremont Street", "Parker Hill"],
        "visual": (
            "A mix of attached 3-4 story brick rowhouses and wooden "
            "triple-deckers on steep hilly streets. Rowhouses have modest "
            "stoops of 4-6 steps, flat rooflines, some bay windows. "
            "Triple-deckers have flat covered porches on each of 3 floors, "
            "wood siding in faded yellows and greens. Streets run steeply uphill."
        )
    },
    "HYDE PARK": {
        "names": ["Cleary Square", "Fairmount Avenue", "Hyde Park Avenue"],
        "visual": (
            "Predominantly wooden triple-decker houses 3 stories tall with "
            "flat covered porches on each level, wood siding in blues, "
            "yellows and whites, small front yards with low fences. Some "
            "single-family detached wooden houses with driveways. Wide "
            "residential streets with mature trees."
        )
    },
    "ROSLINDALE": {
        "names": ["Roslindale Square", "Belgrade Avenue", "Adams Street"],
        "visual": (
            "Wooden triple-decker houses 3 stories tall on most residential "
            "streets, flat covered porches on each level, wood siding in "
            "varied colors, small front yards with fences. Main commercial "
            "square has 2-3 story brick buildings with plain awnings."
        )
    },
    "MATTAPAN": {
        "names": ["Mattapan Square", "Blue Hill Avenue", "River Street"],
        "visual": (
            "Streets lined with wooden triple-decker houses 3 stories tall, "
            "flat covered porches on each floor, wood siding in yellows, "
            "greens, blues, small front yards with chain-link fences. "
            "Wide commercial avenue with 2-story flat-roofed brick buildings, "
            "plain solid-color awnings, no readable text."
        )
    },
    "JAMAICA PLAIN": {
        "names": ["Jamaica Pond", "Centre Street", "Arnold Arboretum"],
        "visual": (
            "Tree-lined streets with a mix of wooden triple-decker houses "
            "3 stories tall with flat covered porches, painted in blues, "
            "greens and yellows — and some attached 2-3 story brick "
            "rowhouses on certain blocks with modest stoops. Centre Street "
            "commercial strip has independent coffee shops and Latin "
            "restaurants in low 2-story brick storefronts with plain "
            "solid-color awnings, no readable text, outdoor seating."
        )
    },
    "DORCHESTER": {
        "names": ["Dorchester Avenue", "Fields Corner", "Columbia Road"],
        "visual": (
            "Streets dominated by wooden triple-decker houses 3 stories "
            "tall, flat covered porches on each floor, wood siding in "
            "faded yellows, greens and blues, small front yards with "
            "chain-link fences. Wide commercial avenue with 2-3 story "
            "flat-roofed brick buildings, plain solid-color awnings, "
            "no readable text. MBTA red line station entrance headhouse "
            "on the sidewalk."
        )
    },
    "DOWNTOWN": {
        "names": ["Faneuil Hall", "Quincy Market", "Government Center"],
        "visual": (
            "A historic 3-story red brick building with a white cupola "
            "and copper grasshopper weathervane on top, surrounded by "
            "a long low granite colonnade building with arched doorways. "
            "Wide open brick plaza with street performers, tourists, "
            "food carts with umbrellas. Mix of 4-story historic brick "
            "buildings and modern glass towers behind."
        )
    },
    "CAMBRIDGE": {
        "names": ["Harvard Yard", "Harvard Square", "Massachusetts Avenue"],
        "visual": (
            "Red brick Georgian and Federal-style academic buildings 3-4 "
            "stories tall with white cupolas, set behind black wrought "
            "iron fences with a large open brick courtyard with massive "
            "oak trees visible through the gates. Harvard Square has a "
            "busy pedestrian plaza with a historic round newsstand kiosk, "
            "bookshops and cafes with plain facades and outdoor seating, "
            "students with backpacks. No readable text anywhere."
        )
    },
    "SOMERVILLE": {
        "names": ["Davis Square", "Union Square", "Prospect Hill"],
        "visual": (
            "Residential streets lined with wooden triple-decker houses "
            "3 stories tall with flat covered porches, painted in greens, "
            "yellows, blues, parked bicycles chained to fences. Main "
            "commercial square has 2-3 story brick buildings with "
            "independent restaurants and bars, plain solid-color awnings, "
            "no readable text. MBTA station entrance with a red line "
            "headhouse on the sidewalk."
        )
    },
    "SOUTH BOSTON": {
        "names": ["Castle Island", "L Street Beach", "East Broadway"],
        "visual": (
            "Dense rows of attached 3-4 story red brick rowhouses with "
            "rounded bay windows on every floor running the full length "
            "of the block. Each rowhouse has a modest front stoop of 4-6 "
            "brick steps with simple iron railings, flat rooflines, iron "
            "fire escapes on the facades. Wide straight streets with brick "
            "sidewalks, overhead utility wires, parked cars bumper to bumper "
            "on both sides. Working-class Irish-American character — plain "
            "brick facades, no ornate trim, some corner bars and small "
            "shops with plain solid-color awnings at ground level. "
            "No readable text anywhere."
        )
    },
    "MEDFORD": {
        "names": ["Tufts University", "Chevalier Theatre", "Medford Square"],
        "visual": (
            "A suburban Greater Boston street lined with classic New England "
            "triple-decker houses. Each house is a detached 3-story wooden "
            "structure standing alone on its own lot with a narrow driveway "
            "on one side and a small front yard with a chain-link or wooden "
            "fence. Each of the 3 floors has a flat covered porch running "
            "the full width of the house with simple wooden railings, the "
            "porches stacked directly above each other. Wood clapboard siding "
            "painted in faded greens, yellows, and whites with some grey and "
            "beige. Moderate-width residential street with mature oak and "
            "maple trees, concrete sidewalks, utility poles with overhead "
            "wires, parked cars along both sides. The houses are NOT attached "
            "— each sits separately on its own lot. NO bay windows, NO ornate "
            "Victorian trim, NO brownstones, NO brick rowhouses. Pure working-"
            "class New England suburban residential street."
        )
    },
    "ALLSTON": {
        "names": ["Brighton Avenue", "Harvard Avenue", "Allston Village"],
        "visual": (
            "Dense college neighborhood streets lined with wooden "
            "triple-decker houses 3 stories tall with flat porches on "
            "each level, wood siding in faded colors, small front yards "
            "packed with bicycles. Some 3-4 story brick buildings mixed "
            "in on busier streets. Ground floor commercial strip has small "
            "restaurants and shops with plain awnings, no readable text."
        )
    }
}

# ── Landmarks Cache ────────────────────────────────────────────────────────────
def load_landmarks_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_landmarks_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def get_cached_landmarks(neighborhood: str) -> dict:
    cache = load_landmarks_cache()
    key = neighborhood.upper()
    if key in cache:
        print(f"[Image] Cache hit for '{neighborhood}' — skipping Tavily + Cortex.")
        return cache[key]
    return {}

def cache_landmarks(neighborhood: str, data: dict):
    cache = load_landmarks_cache()
    cache[neighborhood.upper()] = data
    save_landmarks_cache(cache)
    print(f"[Image] Cached landmarks for '{neighborhood}'.")

# ── Snowflake helpers ──────────────────────────────────────────────────────────
def get_snowflake_conn():
    return snowflake.connector.connect(
        account   = SNOWFLAKE_ACCOUNT,
        user      = SNOWFLAKE_USER,
        password  = SNOWFLAKE_PASSWORD,
        warehouse = SNOWFLAKE_WAREHOUSE,
        database  = SNOWFLAKE_DATABASE,
        role      = SNOWFLAKE_ROLE,
        schema    = "MARTS"
    )

def run_query(sql: str) -> pd.DataFrame:
    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()

def _get_validator():
    """Get a UniversalValidator instance with a fresh Snowflake connection."""
    conn = get_snowflake_conn()
    return UniversalValidator(conn), conn

def get_neighborhood_narrative(neighborhood: str) -> dict:
    sql = f"""
        SELECT
            ml.NEIGHBORHOOD_NAME,
            ml.CITY,
            ml.SQMILES,
            ms.SAFETY_SCORE,
            ms.SAFETY_GRADE,
            ms.SAFETY_DESCRIPTION,
            mr.RESTAURANT_SCORE,
            mr.RESTAURANT_DESCRIPTION,
            mt.TRANSIT_SCORE,
            mt.TRANSIT_GRADE,
            mt.RAPID_TRANSIT_ROUTES,
            mt.BUS_ROUTES,
            mt.TRANSIT_DESCRIPTION,
            mu.EDUCATION_SCORE,
            mu.EDUCATION_DESCRIPTION
        FROM MARTS.MASTER_LOCATION ml
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_SAFETY ms
            ON ml.LOCATION_ID = ms.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_RESTAURANTS mr
            ON ml.LOCATION_ID = mr.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_MBTA mt
            ON ml.LOCATION_ID = mt.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_UNIVERSITIES mu
            ON ml.LOCATION_ID = mu.LOCATION_ID
        WHERE UPPER(ml.NEIGHBORHOOD_NAME) = '{neighborhood.upper()}'
        LIMIT 1
    """
    df = run_query(sql)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

# ── Cortex router ──────────────────────────────────────────────────────────────
SCHEMA_CONTEXT = """
Available Snowflake marts in NEIGHBOURWISE_DOMAINS.MARTS:

MASTER_LOCATION(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, SQMILES, IS_BOSTON, IS_CAMBRIDGE, IS_GREATER_BOSTON)
  - IS_GREATER_BOSTON=TRUE for: Somerville, Arlington, Brookline, Newton, Watertown, Medford, Malden, Revere, Chelsea, Everett, Salem, Quincy
  - IS_BOSTON=TRUE for the 26 Boston neighborhoods
  - IS_CAMBRIDGE=TRUE for the 13 Cambridge neighborhoods
  - ALWAYS join to MASTER_LOCATION using LOCATION_ID to filter by IS_BOSTON/IS_CAMBRIDGE/IS_GREATER_BOSTON

MRT_NEIGHBORHOOD_SAFETY(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, SAFETY_SCORE, SAFETY_GRADE, TOTAL_INCIDENTS, PCT_VIOLENT, PCT_PROPERTY, CRIME_DENSITY_PER_SQMILE, SAFETY_DESCRIPTION)
MRT_NEIGHBORHOOD_MBTA(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, TRANSIT_SCORE, TRANSIT_GRADE, TOTAL_STOPS, RAPID_TRANSIT_ROUTES, BUS_ROUTES, TRANSIT_DESCRIPTION)
MRT_NEIGHBORHOOD_BLUEBIKES(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, BIKESHARE_SCORE, BIKESHARE_GRADE, TOTAL_STATIONS, TOTAL_DOCKS)
MRT_NEIGHBORHOOD_SCHOOLS(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, SCHOOL_SCORE, SCHOOL_GRADE, TOTAL_SCHOOLS)
MRT_NEIGHBORHOOD_RESTAURANTS(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, RESTAURANT_SCORE, RESTAURANT_GRADE, TOTAL_RESTAURANTS, AVG_RATING, RESTAURANT_DESCRIPTION)
MRT_NEIGHBOURHOOD_GROCERY_STORES(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, GROCERY_SCORE, GROCERY_GRADE, TOTAL_STORES)
MRT_NEIGHBORHOOD_HEALTHCARE(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, HEALTHCARE_SCORE, HEALTHCARE_GRADE)
MRT_NEIGHBORHOOD_UNIVERSITIES(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, EDUCATION_SCORE, EDUCATION_GRADE, TOTAL_UNIVERSITIES, EDUCATION_DESCRIPTION)
MRT_NEIGHBORHOOD_HOUSING(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, HOUSING_SCORE, HOUSING_GRADE)
ANALYTICS.NEIGHBORHOOD_MASTER_SCORE(LOCATION_ID, NEIGHBORHOOD_NAME, CITY, MASTER_SCORE, MASTER_GRADE, SAFETY_SCORE, TRANSIT_SCORE, HOUSING_SCORE, GROCERY_SCORE, HEALTHCARE_SCORE, SCHOOL_SCORE, RESTAURANT_SCORE, EDUCATION_SCORE, TOP_STRENGTH, TOP_WEAKNESS)
  *** CRITICAL: This table does NOT have IS_BOSTON, IS_CAMBRIDGE, IS_GREATER_BOSTON columns ***
  To filter by city group: JOIN MARTS.MASTER_LOCATION ml ON nms.LOCATION_ID = ml.LOCATION_ID WHERE ml.IS_BOSTON = TRUE
  To filter by city name:  WHERE UPPER(nms.CITY) = 'BOSTON'
  NEVER write: WHERE IS_BOSTON = TRUE directly on NEIGHBORHOOD_MASTER_SCORE — this will crash.

IMPORTANT SQL RULES:
1. IS_BOSTON, IS_CAMBRIDGE, IS_GREATER_BOSTON ONLY exist on MASTER_LOCATION — never use them on any other table
2. To filter by city group on any domain mart, JOIN to MASTER_LOCATION:
   JOIN MARTS.MASTER_LOCATION ml ON nms.LOCATION_ID = ml.LOCATION_ID WHERE ml.IS_GREATER_BOSTON = TRUE
3. To filter by a single city use WHERE UPPER(CITY) = 'CITYNAME' directly on any mart — no JOIN needed
4. All scores are 0-100, higher = better
5. Use fully qualified table names: MARTS.MRT_NEIGHBORHOOD_SAFETY etc.
6. For ANALYTICS schema: NEIGHBOURWISE_DOMAINS.ANALYTICS.NEIGHBORHOOD_MASTER_SCORE
7. NEIGHBORHOOD_NAME values are UPPER CASE — always use UPPER() when filtering:
   WHERE UPPER(ml.NEIGHBORHOOD_NAME) IN ('JAMAICA PLAIN', 'DORCHESTER')
8. CITY values are UPPER CASE — always use UPPER() when filtering:
   WHERE UPPER(CITY) = 'BOSTON'
9. CRITICAL — Default scope rules:
   - If the user says "Boston neighborhoods" → filter IS_BOSTON = TRUE
   - If the user says "Cambridge neighborhoods" → filter IS_CAMBRIDGE = TRUE
   - If the user says "suburbs" or "Greater Boston" → filter IS_GREATER_BOSTON = TRUE
   - If no location filter → query ALL 51 locations with NO WHERE filter
   - Only apply location filters when user EXPLICITLY mentions a specific area
10. SQL ALIAS RULES — use these exact aliases:
   - MASTER_LOCATION → ml
   - MRT_NEIGHBORHOOD_SAFETY → ns
   - MRT_NEIGHBORHOOD_MBTA → nm
   - MRT_NEIGHBORHOOD_SCHOOLS → nsc
   - MRT_NEIGHBORHOOD_RESTAURANTS → nr
   - MRT_NEIGHBOURHOOD_GROCERY_STORES → ng
   - MRT_NEIGHBORHOOD_HEALTHCARE → nh
   - MRT_NEIGHBORHOOD_HOUSING → nho
   - MRT_NEIGHBORHOOD_BLUEBIKES → nb
   - MRT_NEIGHBORHOOD_UNIVERSITIES → nu
   - ANALYTICS.NEIGHBORHOOD_MASTER_SCORE → nms
11. NULL FILTERING — ALWAYS add IS NOT NULL filters for every score column in WHERE clause:
    Example: WHERE ns.SAFETY_SCORE IS NOT NULL AND ml.IS_BOSTON = TRUE
    This prevents Harbor Islands and other NULL-score rows from consuming LIMIT slots.
    For multi-join queries add IS NOT NULL for the primary score column being ranked.
12. CAMBRIDGE CITY NOTE: 'Cambridge' is NOT a valid NEIGHBORHOOD_NAME.
    Cambridge is a city. These are the EXACT Cambridge neighborhood names
    that have universities (from actual database):
    - 'AREA 2/MIT' (ID 37) → Massachusetts Institute of Technology
    - 'NEIGHBORHOOD NINE' (ID 28) → Harvard Law School, Harvard-Smithsonian,
      Longy School of Music
    - 'BALDWIN' (ID 39) → Lesley University, Harvard Graduate School of Design
    - 'EAST CAMBRIDGE' (ID 38) → Hult International Business School

    All 13 Cambridge neighborhood EXACT names:
    AGASSIZ, AREA 2/MIT, BALDWIN, CAMBRIDGEPORT, EAST CAMBRIDGE,
    MID-CAMBRIDGE, NEIGHBORHOOD NINE, NORTH CAMBRIDGE, PEABODY,
    PORT, RIVERSIDE, STRAWBERRY HILL, TOBIN, WEST CAMBRIDGE.

    When user says "Cambridge" substitute based on context:
    - Students/university/education → use 'AREA 2/MIT' (MIT campus)
    - Research/prestigious university → use 'NEIGHBORHOOD NINE' (Harvard)
    - Young professionals/tech/startup → use 'EAST CAMBRIDGE' (Kendall Square)
    - Families/residential → use 'WEST CAMBRIDGE'
    - General/default → use 'MID-CAMBRIDGE'
    NEVER use 'CAMBRIDGEPORT' for student queries — zero universities.
    NEVER filter WHERE NEIGHBORHOOD_NAME = 'CAMBRIDGE' — returns no rows.
    NEVER use 'MIT' as a neighborhood name — the correct name is 'AREA 2/MIT'.
"""

SEMANTIC_INTENT_MAPPING = """
When the user mentions these keywords, prioritize these score columns:

FAMILIES / KIDS / CHILDREN / RAISE A FAMILY:
  → Pull: NEIGHBORHOOD_NAME, SCHOOL_SCORE, SAFETY_SCORE, HOUSING_SCORE, GROCERY_SCORE, HEALTHCARE_SCORE, TRANSIT_SCORE
  → Order by: SCHOOL_SCORE DESC
  → WHERE: SCHOOL_SCORE IS NOT NULL AND SAFETY_SCORE IS NOT NULL
  → chart_type: "grouped_bar"

STUDENTS / COLLEGE / UNIVERSITY / YOUNG ADULTS:
  → Pull: NEIGHBORHOOD_NAME, EDUCATION_SCORE, TRANSIT_SCORE, RESTAURANT_SCORE, SAFETY_SCORE, HOUSING_SCORE
  → Order by: EDUCATION_SCORE DESC
  → WHERE: EDUCATION_SCORE IS NOT NULL
  → chart_type: "grouped_bar"

YOUNG PROFESSIONALS / WORK / COMMUTE / CAREER:
  → Pull: NEIGHBORHOOD_NAME, TRANSIT_SCORE, SAFETY_SCORE, RESTAURANT_SCORE, HOUSING_SCORE, MASTER_SCORE
  → Order by: TRANSIT_SCORE DESC
  → WHERE: TRANSIT_SCORE IS NOT NULL
  → chart_type: "grouped_bar"

RETIREES / SENIORS / QUIET / PEACEFUL:
  → Pull: NEIGHBORHOOD_NAME, SAFETY_SCORE, HEALTHCARE_SCORE, GROCERY_SCORE, TRANSIT_SCORE, HOUSING_SCORE
  → Order by: SAFETY_SCORE DESC
  → WHERE: SAFETY_SCORE IS NOT NULL
  → chart_type: "grouped_bar"

FOODIES / RESTAURANTS / NIGHTLIFE / GOING OUT / FOOD:
  → Pull: NEIGHBORHOOD_NAME, RESTAURANT_SCORE, GROCERY_SCORE, TRANSIT_SCORE, SAFETY_SCORE
  → Order by: RESTAURANT_SCORE DESC
  → WHERE: RESTAURANT_SCORE IS NOT NULL
  → chart_type: "bar"

SAFEST / SAFETY / CRIME / DANGEROUS / SECURE:
  → Pull: NEIGHBORHOOD_NAME, SAFETY_SCORE, SAFETY_GRADE
  → Order by: SAFETY_SCORE DESC
  → WHERE: SAFETY_SCORE IS NOT NULL
  → Source: MARTS.MRT_NEIGHBORHOOD_SAFETY joined to MASTER_LOCATION
  → chart_type: "bar"

OVERALL / BEST / TOP / RANK / COMPARE ALL / MASTER:
  → Pull: NEIGHBORHOOD_NAME, MASTER_SCORE, MASTER_GRADE, TOP_STRENGTH, TOP_WEAKNESS
  → Order by: MASTER_SCORE DESC
  → WHERE: MASTER_SCORE IS NOT NULL
  → Source: ANALYTICS.NEIGHBORHOOD_MASTER_SCORE (alias: nms)
  → chart_type: "bar"
  → To filter Boston only: JOIN MARTS.MASTER_LOCATION ml ON nms.LOCATION_ID = ml.LOCATION_ID WHERE ml.IS_BOSTON = TRUE AND nms.MASTER_SCORE IS NOT NULL
  → To filter Boston + Greater Boston: JOIN MARTS.MASTER_LOCATION ml ON nms.LOCATION_ID = ml.LOCATION_ID WHERE (ml.IS_BOSTON = TRUE OR ml.IS_GREATER_BOSTON = TRUE) AND nms.MASTER_SCORE IS NOT NULL
  → To filter by city name directly (no join needed): WHERE UPPER(nms.CITY) = 'BOSTON' AND nms.MASTER_SCORE IS NOT NULL
  → NEVER write IS_BOSTON = TRUE without the MASTER_LOCATION JOIN

TRANSIT / COMMUTE / MBTA / BUS / SUBWAY / TRANSPORT:
  → Pull: NEIGHBORHOOD_NAME, TRANSIT_SCORE, RAPID_TRANSIT_ROUTES, BUS_ROUTES, TRANSIT_GRADE
  → Order by: TRANSIT_SCORE DESC
  → WHERE: TRANSIT_SCORE IS NOT NULL
  → Source: MARTS.MRT_NEIGHBORHOOD_MBTA joined to MASTER_LOCATION
  → chart_type: "bar"

AFFORDABLE / CHEAP / BUDGET / HOUSING / COST / PRICE:
  → Pull: NEIGHBORHOOD_NAME, HOUSING_SCORE, HOUSING_GRADE, SAFETY_SCORE, MASTER_SCORE
  → Order by: HOUSING_SCORE DESC
  → WHERE: HOUSING_SCORE IS NOT NULL
  → chart_type: "bar"

SCHOOLS / EDUCATION / LEARNING:
  → Pull: NEIGHBORHOOD_NAME, SCHOOL_SCORE, SCHOOL_GRADE, TOTAL_SCHOOLS
  → Order by: SCHOOL_SCORE DESC
  → WHERE: SCHOOL_SCORE IS NOT NULL
  → Source: MARTS.MRT_NEIGHBORHOOD_SCHOOLS joined to MASTER_LOCATION
  → chart_type: "bar"

RELATIONSHIP / CORRELATION / DOES X AFFECT / VS between two metrics:
  → Pull both metrics for all 51 neighborhoods — no LIMIT
  → WHERE: both score columns IS NOT NULL
  → chart_type: "scatter"
  → include NEIGHBORHOOD_NAME as third column

TREND / OVER TIME / MONTHLY / BY SEASON / HOW HAS X CHANGED:
  → Pull time-series data from MRT_BOSTON_CRIME
  → If comparing ONE neighborhood → chart_type: "line"
  → If comparing TWO or MORE neighborhoods over time → chart_type: "multi_line"
  → For multi_line: SELECT YEAR_MONTH, NEIGHBORHOOD_NAME, COUNT(*) as CRIME_COUNT
    FROM MARTS.MRT_BOSTON_CRIME
    WHERE UPPER(NEIGHBORHOOD_NAME) IN ('NEIGHBORHOOD1', 'NEIGHBORHOOD2')
    AND YEAR = YYYY
    GROUP BY YEAR_MONTH, NEIGHBORHOOD_NAME
    ORDER BY YEAR_MONTH ASC
  → x axis = YEAR_MONTH, y axis = count/metric, color = NEIGHBORHOOD_NAME

PERCENTAGE / BREAKDOWN / PROPORTION / WHAT FRACTION / HOW MANY ARE:
  → Pull counts grouped by grade or category
  → WHERE: grade IS NOT NULL
  → chart_type: "pie"

For TWO neighborhood comparisons — always use chart_type "grouped_bar" and pull
ALL relevant domain scores not just MASTER_SCORE.
NEVER include MASTER_SCORE in a grouped_bar domain comparison — it is a composite
of the other scores and will distort the visual. Only include the 9 individual
domain scores: SAFETY, TRANSIT, SCHOOL, RESTAURANT, GROCERY, HEALTHCARE,
HOUSING, BIKESHARE, EDUCATION.

LIMIT RULES — CRITICAL:
- Default: ALWAYS add LIMIT 5 unless user says otherwise
- "top 10" → LIMIT 10
- "top 3" → LIMIT 3
- "all neighborhoods" / "all 51" / "rank all" → NO limit
- "compare X and Y" (two specific neighborhoods) → NO limit, just those 2
- scatter plots → NO limit
- Never return more than 15 rows unless explicitly asked
"""

def classify_intent(user_query: str, correction_suffix: str = "") -> dict:
    prompt = f"""You are a routing agent for NeighbourWise AI, a Boston neighborhood app.

Classify this query and respond with ONLY a JSON object (no markdown, no explanation):

Query: "{user_query}"

Rules:
- chart/graph/plot/trend/comparison/ranking/visualization → intent = "chart"
- image/photo/picture/show me what X looks like / show me [neighborhood] → intent = "image"
- Any comparison, recommendation, or analytical question → intent = "chart"
- For "chart": write Snowflake SQL to fetch the data
- For "image": extract the neighborhood name

CHART TYPE SELECTION RULES:
- "relationship", "correlation", "does X affect Y", "vs" between two metrics → "scatter"
- "trend", "over time", "monthly", "by season", "how has X changed"
  with ONE neighborhood → "line"
- "trend", "over time", "monthly" comparing TWO or MORE neighborhoods → "multi_line"
- "percentage", "breakdown", "proportion", "what fraction", "how many are" → "pie"
- "compare X and Y across domains", "better for [persona]" with 2+ neighborhoods → "grouped_bar"
- "which is best", "top", "rank", "safest", "highest", "lowest", single metric → "bar"

CRITICAL FOR SQL GENERATION:
- Use SEMANTIC_INTENT_MAPPING to decide WHICH score columns to pull
- ALWAYS add IS NOT NULL filter for every score column used in WHERE/ORDER BY
- For comparison queries pull ALL relevant domain scores not just MASTER_SCORE
- ALWAYS use UPPER() when filtering by NEIGHBORHOOD_NAME or CITY
- ALWAYS apply LIMIT 5 by default unless query asks for more
- Use exact table aliases from SQL ALIAS RULES to avoid conflicts
- CRITICAL SCOPE RULE: If the query does NOT explicitly mention "Boston",
  "Cambridge", "suburbs", or "Greater Boston" — query ALL 51 neighborhoods
  with NO IS_BOSTON / IS_CAMBRIDGE / IS_GREATER_BOSTON filter whatsoever.
  Only apply city filters when the user explicitly asks for a specific city.

{SCHEMA_CONTEXT}

{SEMANTIC_INTENT_MAPPING}
{correction_suffix}

JSON structure:
{{
  "intent": "chart" or "image",
  "chart_type": "bar" or "line" or "scatter" or "pie" or "grouped_bar" or null,
  "title": "descriptive chart title" or null,
  "x_label": "x axis label" or null,
  "y_label": "y axis label" or null,
  "sql": "SELECT ... FROM ... WHERE [score] IS NOT NULL ... ORDER BY ... [LIMIT N]" or null,
  "neighborhood": "neighborhood name" or null,
  "reasoning": "one sentence explanation"
}}"""

    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()
        safe_prompt = prompt.replace("'", "\\'")
        cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{safe_prompt}')")
        result = cur.fetchone()[0]
        result = re.sub(r'```json|```', '', result).strip()
        return json.loads(result)
    except Exception as e:
        print(f"[Cortex router error] {e}")
        return {"intent": "unknown", "reasoning": str(e)}
    finally:
        conn.close()

# ── SQL Post-processors ────────────────────────────────────────────────────────
def fix_sql_scope(sql: str, user_query: str) -> str:
    query_lower = user_query.lower()
    all_keywords = [
        "where should i live", "best neighborhood", "top neighborhood",
        "recommend", "good food and transit", "good for families",
        "good for students", "good for young", "good for retirees",
        "where can i", "which neighborhood", "which neighborhoods",
        "best place", "relationship between", "correlation", "does",
        "top 5", "top 10", "top 3", "safest", "show me the top",
        "rank", "highest", "lowest", "most", "least",
    ]
    specific_keywords = [
        "boston neighborhood", "in boston", "cambridge neighborhood",
        "in cambridge", "greater boston", "suburbs",
    ]
    wants_specific = any(k in query_lower for k in specific_keywords)
    wants_all      = any(k in query_lower for k in all_keywords)

    if wants_all and not wants_specific:
        sql = re.sub(r"\bml\.IS_GREATER_BOSTON\s*=\s*TRUE\b", "1=1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bml\.IS_BOSTON\s*=\s*TRUE\b",         "1=1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bml\.IS_CAMBRIDGE\s*=\s*TRUE\b",      "1=1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bAND\s+1=1\b",   "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\b1=1\s+AND\b",   "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\bWHERE\s+1=1\b", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s+", " ", sql).strip()
        print(f"[Chart] Scope fix applied.")
    return sql

def fix_sql_ordering(sql: str, user_query: str) -> str:
    query_lower = user_query.lower()
    combined = [
        "food and transit", "transit and food", "food and safety",
        "safety and food", "transit and safety", "safety and transit",
        "good for young professional", "good for families",
        "good for students", "balance", "both",
    ]
    if any(k in query_lower for k in combined):
        fixed = re.sub(
            r"ORDER BY\s+(\w+)\s+DESC\s*,\s*(\w+)\s+DESC",
            r"ORDER BY (\1 + \2) / 2 DESC",
            sql, flags=re.IGNORECASE
        )
        if fixed != sql:
            print(f"[Chart] Ordering fix applied.")
            return fixed
    return sql

def fix_sql_aliases(sql: str) -> str:
    replacements = [
        (r'\bMRT_NEIGHBORHOOD_SCHOOLS\s+s\b',              'MRT_NEIGHBORHOOD_SCHOOLS nsc'),
        (r'\bMRT_NEIGHBORHOOD_SAFETY\s+s\b',               'MRT_NEIGHBORHOOD_SAFETY ns'),
        (r'\bMRT_NEIGHBORHOOD_RESTAURANTS\s+r\b',          'MRT_NEIGHBORHOOD_RESTAURANTS nr'),
        (r'\bMRT_NEIGHBOURHOOD_GROCERY_STORES\s+g\b',      'MRT_NEIGHBOURHOOD_GROCERY_STORES ng'),
        (r'\bMRT_NEIGHBORHOOD_GROCERY_STORES\s+g\b',       'MRT_NEIGHBOURHOOD_GROCERY_STORES ng'),
        (r'\bMRT_NEIGHBORHOOD_GROCERY\s+g\b',              'MRT_NEIGHBOURHOOD_GROCERY_STORES ng'),
        (r'\bMRT_NEIGHBORHOOD_HEALTHCARE\s+h\b',           'MRT_NEIGHBORHOOD_HEALTHCARE nh'),
        (r'\bMRT_NEIGHBOURHOOD_HOUSING\s+h\b',             'MRT_NEIGHBORHOOD_HOUSING nho'),
        (r'\bMRT_NEIGHBORHOOD_HOUSING\s+h\b',              'MRT_NEIGHBORHOOD_HOUSING nho'),
        (r'\bMRT_NEIGHBORHOOD_MBTA\s+t\b',                 'MRT_NEIGHBORHOOD_MBTA nm'),
        (r'\bMRT_NEIGHBORHOOD_BLUEBIKES\s+b\b',            'MRT_NEIGHBORHOOD_BLUEBIKES nb'),
        (r'\bMRT_NEIGHBORHOOD_UNIVERSITIES\s+u\b',         'MRT_NEIGHBORHOOD_UNIVERSITIES nu'),
        (r'\bNEIGHBORHOOD_MASTER_SCORE\s+n\b',            'NEIGHBORHOOD_MASTER_SCORE nms'),
    ]
    fixed = sql
    for pattern, replacement in replacements:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

    col_fixes = [
        (r'\bs\.SAFETY_SCORE\b',         'ns.SAFETY_SCORE'),
        (r'\bs\.SAFETY_GRADE\b',         'ns.SAFETY_GRADE'),
        (r'\bs\.TOTAL_INCIDENTS\b',      'ns.TOTAL_INCIDENTS'),
        (r'\bs\.PCT_VIOLENT\b',          'ns.PCT_VIOLENT'),
        (r'\bs\.SCHOOL_SCORE\b',         'nsc.SCHOOL_SCORE'),
        (r'\bs\.SCHOOL_GRADE\b',         'nsc.SCHOOL_GRADE'),
        (r'\bs\.TOTAL_SCHOOLS\b',        'nsc.TOTAL_SCHOOLS'),
        (r'\br\.RESTAURANT_SCORE\b',     'nr.RESTAURANT_SCORE'),
        (r'\br\.RESTAURANT_GRADE\b',     'nr.RESTAURANT_GRADE'),
        (r'\bg\.GROCERY_SCORE\b',        'ng.GROCERY_SCORE'),
        (r'\bg\.GROCERY_GRADE\b',        'ng.GROCERY_GRADE'),
        (r'\bh\.HEALTHCARE_SCORE\b',     'nh.HEALTHCARE_SCORE'),
        (r'\bh\.HEALTHCARE_GRADE\b',     'nh.HEALTHCARE_GRADE'),
        (r'\bhe\.HEALTHCARE_SCORE\b',    'nh.HEALTHCARE_SCORE'),
        (r'\bhe\.HEALTHCARE_GRADE\b',    'nh.HEALTHCARE_GRADE'),
        (r'\bh\.HOUSING_SCORE\b',        'nho.HOUSING_SCORE'),
        (r'\bh\.HOUSING_GRADE\b',        'nho.HOUSING_GRADE'),
        (r'\bt\.TRANSIT_SCORE\b',        'nm.TRANSIT_SCORE'),
        (r'\bt\.TRANSIT_GRADE\b',        'nm.TRANSIT_GRADE'),
        (r'\bt\.BUS_ROUTES\b',           'nm.BUS_ROUTES'),
        (r'\bt\.RAPID_TRANSIT_ROUTES\b', 'nm.RAPID_TRANSIT_ROUTES'),
    ]
    for pattern, replacement in col_fixes:
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

    if fixed != sql:
        print(f"[Chart] Alias fix applied.")
    return fixed

def fix_sql_nulls(sql: str) -> str:
    order_match = re.search(r'ORDER BY\s+([\w.]+)\s+DESC', sql, re.IGNORECASE)
    if not order_match:
        return sql
    order_col = order_match.group(1)
    if "SCORE" not in order_col.upper():
        return sql
    null_check = f"{order_col} IS NOT NULL"
    if null_check.upper() in sql.upper():
        return sql
    if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
        sql = re.sub(r'\bWHERE\b', f'WHERE {null_check} AND',
                     sql, count=1, flags=re.IGNORECASE)
    else:
        sql = re.sub(r'\b(GROUP BY|ORDER BY)\b',
                     f'WHERE {null_check} \\1',
                     sql, count=1, flags=re.IGNORECASE)
    print(f"[Chart] NULL filter added for {order_col}.")
    return sql

def safe_vegalite_to_png(chart, scale=2) -> bytes:
    chart_json = chart.to_json()
    chart_json = re.sub(r'\bNaN\b',       'null', chart_json)
    chart_json = re.sub(r'\bInfinity\b',  '999',  chart_json)
    chart_json = re.sub(r'\b-Infinity\b', '-999', chart_json)
    return vlc.vegalite_to_png(chart_json, scale=scale)

# ── Chart Generator ────────────────────────────────────────────────────────────
def generate_chart(plan: dict, user_query: str) -> str:
    sql        = plan.get("sql")
    chart_type = plan.get("chart_type", "bar")

    if not sql:
        print("[Chart] No SQL generated.")
        return None

    validator, val_conn = _get_validator()
    try:
        # ── CHECKPOINT 1: Pre-execution SQL validation ────────────────────────
        correction_suffix = ""
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n[Validator] Pre-execution check — attempt {attempt}/{MAX_RETRIES}")

            sql = fix_sql_scope(sql, user_query)
            sql = fix_sql_ordering(sql, user_query)
            sql = fix_sql_aliases(sql)
            sql = fix_sql_nulls(sql)
            print(f"[Chart] SQL: {sql}")

            pre_result = validator.validate(
                AgentType.GRAPHIC_CHART,
                {
                    "sql":        sql,
                    "chart_type": chart_type,
                    "df":         None,
                    "out_path":   None,
                    "user_query": user_query,
                    "attempt":    attempt,
                }
            )
            pre_result.print_summary()

            if pre_result.passed:
                break

            if attempt < MAX_RETRIES:
                issues = pre_result.all_issues
                correction_suffix = (
                    "\n\nVALIDATOR CORRECTIONS REQUIRED:\n" +
                    "\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues))
                )
                print(f"[Validator] Retrying classify_intent with correction hints...")
                plan       = classify_intent(user_query, correction_suffix)
                sql        = plan.get("sql", sql)
                chart_type = plan.get("chart_type", chart_type)
                if not sql:
                    print("[Chart] Cortex returned no SQL on retry.")
                    return None
            else:
                print(f"[Validator] Pre-execution: max retries reached — proceeding with best SQL.")

        # ── Execute SQL ───────────────────────────────────────────────────────
        df = run_query(sql)

        # Strip noisy rows
        for col in df.columns:
            try:
                if df[col].dtype == object or str(df[col].dtype) == 'string':
                    mask = df[col].astype(str).str.upper().str.strip().isin(
                        ["INSUFFICIENT DATA", "HARBOR ISLANDS"]
                    )
                    df = df[~mask]
            except Exception:
                pass
        df = df.reset_index(drop=True)

        # ── CHECKPOINT 2: Post-execution data validation ──────────────────────
        correction_suffix = ""
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n[Validator] Post-execution check — attempt {attempt}/{MAX_RETRIES}")

            post_result = validator.validate(
                AgentType.GRAPHIC_CHART,
                {
                    "sql":        sql,
                    "chart_type": chart_type,
                    "df":         df,
                    "out_path":   None,
                    "user_query": user_query,
                    "attempt":    attempt,
                }
            )
            post_result.print_summary()

            if post_result.passed:
                break

            if attempt < MAX_RETRIES:
                issues = post_result.all_issues
                correction_suffix = (
                    "\n\nVALIDATOR CORRECTIONS REQUIRED:\n" +
                    "\n".join(f"{i+1}. {issue}" for i, issue in enumerate(issues))
                )
                print(f"[Validator] Retrying classify_intent with correction hints...")
                plan       = classify_intent(user_query, correction_suffix)
                sql        = plan.get("sql", sql)
                chart_type = plan.get("chart_type", chart_type)
                if not sql:
                    print("[Chart] Cortex returned no SQL on retry.")
                    return None
                sql = fix_sql_scope(sql, user_query)
                sql = fix_sql_ordering(sql, user_query)
                sql = fix_sql_aliases(sql)
                sql = fix_sql_nulls(sql)
                df  = run_query(sql)
                for col in df.columns:
                    try:
                        if df[col].dtype == object or str(df[col].dtype) == 'string':
                            mask = df[col].astype(str).str.upper().str.strip().isin(
                                ["INSUFFICIENT DATA", "HARBOR ISLANDS"]
                            )
                            df = df[~mask]
                    except Exception:
                        pass
                df = df.reset_index(drop=True)
            else:
                print(f"[Validator] Post-execution: max retries reached — rendering with best data.")

        if df.empty:
            print("[Chart] No renderable data after validation.")
            return None

        print(f"[Chart] Rendering {chart_type} chart with {len(df)} rows...")

        title      = plan.get("title", user_query)
        x_label    = plan.get("x_label", df.columns[0])
        y_label    = plan.get("y_label", df.columns[1] if len(df.columns) > 1 else "")
        x_col      = df.columns[0]
        y_col      = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        title = re.sub(r"\bin Greater Boston\b", "", title,
                       flags=re.IGNORECASE).strip().rstrip(" —-,")

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[^a-zA-Z0-9_]', '_', title[:40])
        out_path   = CHARTS_DIR / f"{safe_title}_{timestamp}.png"

        # ── Design tokens ─────────────────────────────────────────────────────
        FONT    = "Arial"
        TEXT    = "#1A1A2E"
        SUBTEXT = "#6B7280"
        GRID    = "#F0F2F5"
        BG      = "#FAFAFA"
        ACCENT  = "#2E86AB"
        PALETTE = [
            "#2E86AB", "#E84855", "#F4A261",
            "#2A9D8F", "#8338EC", "#F9C74F",
            "#06D6A0", "#EF476F"
        ]

        def altair_config():
            return {
                "background"  : BG,
                "font"        : FONT,
                "title"       : {
                    "fontSize"    : 22,
                    "fontWeight"  : "bold",
                    "color"       : TEXT,
                    "anchor"      : "start",
                    "offset"      : 16,
                    "subtitleColor": SUBTEXT,
                    "subtitleFontSize": 13,
                },
                "axis": {
                    "labelColor"    : TEXT,
                    "labelFontSize" : 12,
                    "labelFontWeight": "normal",
                    "titleColor"    : SUBTEXT,
                    "titleFontSize" : 12,
                    "titleFontWeight": "normal",
                    "gridColor"     : GRID,
                    "gridOpacity"   : 1,
                    "domainColor"   : "#E0E0E0",
                    "tickColor"     : "#E0E0E0",
                    "tickSize"      : 5,
                    "labelFont"     : FONT,
                    "titleFont"     : FONT,
                    "labelPadding"  : 8,
                    "titlePadding"  : 10,
                },
                "legend": {
                    "labelColor"    : TEXT,
                    "labelFontSize" : 12,
                    "titleColor"    : SUBTEXT,
                    "titleFontSize" : 11,
                    "labelFont"     : FONT,
                    "titleFont"     : FONT,
                    "orient"        : "bottom",
                    "direction"     : "horizontal",
                    "padding"       : 10,
                    "columnPadding" : 20,
                },
                "view": {
                    "stroke"      : "transparent",
                    "fill"        : BG,
                },
                "mark": {
                    "font": FONT,
                }
            }

        if chart_type == "bar":
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df        = df.dropna(subset=[y_col])
            df        = df.sort_values(y_col, ascending=True)
            mean_val  = float(df[y_col].mean())
            color_scale = alt.Scale(
                domain=[float(df[y_col].min()), float(df[y_col].max())],
                range=["#E74C3C", "#F39C12", "#27AE60"]
            )
            bars = alt.Chart(df).mark_bar(
                cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=28,
            ).encode(
                y=alt.Y(f"{x_col}:N", sort=None,
                        axis=alt.Axis(title=None, labelLimit=220, labelFontWeight="bold", labelFontSize=12)),
                x=alt.X(f"{y_col}:Q",
                        axis=alt.Axis(title=y_label, grid=True, tickCount=5),
                        scale=alt.Scale(domain=[0, float(df[y_col].max()) * 1.18])),
                color=alt.Color(f"{y_col}:Q", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip(f"{x_col}:N", title="Neighborhood"),
                    alt.Tooltip(f"{y_col}:Q", title=y_label, format=".1f"),
                ]
            )
            text = alt.Chart(df).mark_text(
                align="left", dx=6, fontSize=12, fontWeight="bold", color=TEXT, font=FONT,
            ).encode(
                y=alt.Y(f"{x_col}:N", sort=None),
                x=alt.X(f"{y_col}:Q"),
                text=alt.Text(f"{y_col}:Q", format=".1f"),
            )
            mean_rule = alt.Chart(pd.DataFrame({"mean": [mean_val]})).mark_rule(
                color=SUBTEXT, strokeDash=[5, 4], strokeWidth=1.5, opacity=0.6,
            ).encode(x="mean:Q")
            mean_label = alt.Chart(
                pd.DataFrame({"mean": [mean_val], "label": [f"avg {mean_val:.1f}"], "y": [0]})
            ).mark_text(
                align="left", dx=4, dy=-8, fontSize=10, color=SUBTEXT, font=FONT,
            ).encode(x=alt.X("mean:Q"), y=alt.value(0), text="label:N")
            chart = (bars + text + mean_rule + mean_label).properties(
                title=alt.TitleParams(text=title, subtitle="Score range 0–100  •  Higher is better"),
                width=580, height=max(220, len(df) * 50),
            ).configure(**altair_config())
            out_path.write_bytes(safe_vegalite_to_png(chart, scale=2))

        elif chart_type == "grouped_bar":
            score_cols = [c for c in df.columns[1:]
                          if pd.to_numeric(df[c], errors='coerce').notna().any()]
            DOMAIN_LABEL_MAP = {
                "SAFETY":     "Safety",
                "TRANSIT":    "Transit",
                "SCHOOL":     "Schools",
                "RESTAURANT": "Restaurants",
                "GROCERY":    "Grocery",
                "HEALTHCARE": "Healthcare",
                "HOUSING":    "Housing",
                "BIKESHARE":  "Bikeshare",
                "EDUCATION":  "Universities",
            }
            def clean_col(c):
                key = c.replace("_SCORE", "").upper()
                return DOMAIN_LABEL_MAP.get(key, key.replace("_", " ").title())
            df_long = df[[x_col] + score_cols].melt(
                id_vars=x_col, value_vars=score_cols, var_name="Domain", value_name="Score")
            df_long["Domain"] = df_long["Domain"].apply(clean_col)
            df_long["Score"]  = pd.to_numeric(df_long["Score"], errors='coerce')

            primary_score_col = score_cols[0] if score_cols else None
            if primary_score_col:
                valid_neighborhoods = (
                    df[pd.to_numeric(df[primary_score_col], errors='coerce') > 0][x_col]
                    .unique()
                )
                df_long = df_long[df_long[x_col].isin(valid_neighborhoods)]

            domain_list = df_long["Domain"].unique().tolist()
            color_scale = alt.Scale(domain=domain_list, range=PALETTE[:len(domain_list)])

            LABEL_OVERRIDES = {
                "NEIGHBORHOOD NINE":    "NGHBRHD NINE",
                "EAST CAMBRIDGE":       "EAST CAMB.",
                "NORTH CAMBRIDGE":      "NORTH CAMB.",
                "WEST CAMBRIDGE":       "WEST CAMB.",
                "MID-CAMBRIDGE":        "MID-CAMB.",
                "STRAWBERRY HILL":      "STRAW. HILL",
                "CAMBRIDGEPORT":        "CAMB.PORT",
                "JAMAICA PLAIN":        "JAM. PLAIN",
                "MISSION HILL":         "MISS. HILL",
                "HARBOR ISLANDS":       "HARBOR ISL.",
                "BACK BAY":             "BACK BAY",
                "SOUTH END":            "SOUTH END",
                "EAST BOSTON":          "EAST BOSTON",
                "WELLINGTON-HARRINGTON":"WELLINGTON",
            }

            def shorten_label(name: str) -> str:
                upper = str(name).upper().strip()
                if upper in LABEL_OVERRIDES:
                    return LABEL_OVERRIDES[upper]
                if len(upper) > 12:
                    cut = upper[:12].rfind(" ")
                    if cut > 0:
                        return upper[:cut] + "."
                    return upper[:12] + "."
                return upper

            df_long[x_col] = df_long[x_col].apply(shorten_label)

            n_neighborhoods = df_long[x_col].nunique()
            if n_neighborhoods >= 5:
                label_font_size = 10
                col_spacing     = 10
            elif n_neighborhoods == 4:
                label_font_size = 11
                col_spacing     = 14
            else:
                label_font_size = 13
                col_spacing     = 24

            bars = alt.Chart(df_long).mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3, opacity=0.92,
            ).encode(
                x=alt.X("Domain:N", axis=alt.Axis(labels=False, title=None, ticks=False)),
                y=alt.Y("Score:Q", axis=alt.Axis(title="Score (0–100)", grid=True),
                        scale=alt.Scale(domain=[0, 118])),
                color=alt.Color("Domain:N", scale=color_scale, legend=alt.Legend(title="Domain")),
                tooltip=[
                    alt.Tooltip(f"{x_col}:N", title="Neighborhood"),
                    alt.Tooltip("Domain:N", title="Domain"),
                    alt.Tooltip("Score:Q", title="Score", format=".1f"),
                ]
            )
            text = alt.Chart(df_long).mark_text(
                dy=-7, fontSize=10, fontWeight="bold", color=TEXT, font=FONT,
            ).encode(
                x=alt.X("Domain:N"), y=alt.Y("Score:Q"),
                text=alt.Text("Score:Q", format=".0f"),
                color=alt.Color("Domain:N", scale=color_scale, legend=None),
            )
            chart = alt.layer(bars, text).facet(
                column=alt.Column(f"{x_col}:N",
                    header=alt.Header(
                        titleFontSize=0,
                        labelFontSize=label_font_size,
                        labelFontWeight="bold",
                        labelColor=TEXT,
                        labelFont=FONT,
                        labelPadding=10,
                        labelLimit=160,
                    )),
                spacing=col_spacing,
            ).properties(
                title=alt.TitleParams(text=title, subtitle="Score range 0–100  •  Higher is better"),
            ).configure(**altair_config())
            out_path.write_bytes(safe_vegalite_to_png(chart, scale=2))

        elif chart_type == "scatter":
            numeric_cols = []
            label_col    = None
            for col in df.columns:
                converted = pd.to_numeric(df[col], errors='coerce')
                if converted.notna().sum() > len(df) * 0.5:
                    numeric_cols.append(col)
                else:
                    if label_col is None: label_col = col
            if len(numeric_cols) < 2:
                print("[Chart] Not enough numeric columns for scatter.")
                return None
            x_col = numeric_cols[0]; y_col = numeric_cols[1]
            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[x_col, y_col])
            x_label = x_col.replace("_SCORE","").replace("_"," ").title()
            y_label = y_col.replace("_SCORE","").replace("_"," ").title()
            label_col = df.columns[2] if len(df.columns) > 2 else None
            x_clean = df[x_col]; y_clean = df[y_col]
            corr = float(np.corrcoef(x_clean.values, y_clean.values)[0,1]) if len(x_clean)>2 else 0.0
            if np.isnan(corr): corr = 0.0
            direction = "positive" if corr > 0 else "negative"
            strength = "strong" if abs(corr)>0.6 else "moderate" if abs(corr)>0.3 else "weak"
            x_pad = (float(df[x_col].max())-float(df[x_col].min()))*0.05
            y_pad = (float(df[y_col].max())-float(df[y_col].min()))*0.05
            x_domain = [round(float(df[x_col].min())-x_pad,2), round(float(df[x_col].max())+x_pad,2)]
            y_domain = [round(float(df[y_col].min())-y_pad,2), round(float(df[y_col].max())+y_pad,2)]
            y_min = round(float(y_clean.min()),2); y_max = round(float(y_clean.max()),2)
            color_scale = alt.Scale(domain=[y_min,y_max], range=["#E74C3C","#F39C12","#27AE60"])
            encode_kwargs = dict(
                x=alt.X(f"{x_col}:Q", scale=alt.Scale(domain=x_domain), axis=alt.Axis(title=x_label,grid=True)),
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=y_domain), axis=alt.Axis(title=y_label,grid=True)),
                color=alt.Color(f"{y_col}:Q", scale=color_scale,
                    legend=alt.Legend(title=f"{y_label} Score", titleFontSize=11,
                        titleColor=SUBTEXT, labelFontSize=10, gradientLength=120,
                        gradientThickness=12, orient="right")),
                tooltip=([alt.Tooltip(f"{label_col}:N",title="Neighborhood"),
                    alt.Tooltip(f"{x_col}:Q",title=x_label,format=".1f"),
                    alt.Tooltip(f"{y_col}:Q",title=y_label,format=".1f")]
                    if label_col else
                    [alt.Tooltip(f"{x_col}:Q",title=x_label,format=".1f"),
                     alt.Tooltip(f"{y_col}:Q",title=y_label,format=".1f")])
            )
            points = alt.Chart(df).mark_circle(size=100,opacity=0.88,stroke="white",strokeWidth=1.8).encode(**encode_kwargs)
            if len(x_clean)>2:
                z=np.polyfit(x_clean.values,y_clean.values,1); p=np.poly1d(z)
                x_line=np.linspace(float(x_clean.min()),float(x_clean.max()),50)
                trend_df=pd.DataFrame({x_col:[round(float(v),4) for v in x_line],y_col:[round(float(v),4) for v in p(x_line)]})
                trend=alt.Chart(trend_df).mark_line(color=ACCENT,strokeDash=[6,3],strokeWidth=2,opacity=0.55).encode(x=alt.X(f"{x_col}:Q"),y=alt.Y(f"{y_col}:Q"))
            else: trend=None
            if label_col and len(df)>0:
                df=df.copy(); df["_dist"]=((df[x_col]-float(df[x_col].max()))**2+(df[y_col]-float(df[y_col].max()))**2)
                label_idx=list(set(list(df.nsmallest(5,"_dist").index)+list(df.nlargest(3,"_dist").index)))
                labels=alt.Chart(df.loc[label_idx].copy()).mark_text(dy=-13,fontSize=9.5,fontWeight="bold",font=FONT,color=TEXT).encode(x=alt.X(f"{x_col}:Q"),y=alt.Y(f"{y_col}:Q"),text=alt.Text(f"{label_col}:N"))
            else: labels=None
            layers=[points]
            if trend: layers.insert(0,trend)
            if labels: layers.append(labels)
            chart=alt.layer(*layers).properties(
                title=alt.TitleParams(text=title,subtitle=f"r = {corr:.2f}  |  {strength} {direction} correlation"),
                width=640,height=460).configure(**altair_config())
            out_path.write_bytes(safe_vegalite_to_png(chart, scale=2))

        elif chart_type == "line":
            df[y_col]=pd.to_numeric(df[y_col],errors='coerce'); df=df.dropna(subset=[y_col]); y_vals=df[y_col].values
            area=alt.Chart(df).mark_area(
                line={"color":ACCENT,"strokeWidth":2.8},
                color=alt.Gradient(gradient="linear",stops=[alt.GradientStop(color="rgba(46,134,171,0.28)",offset=0),alt.GradientStop(color="rgba(46,134,171,0.02)",offset=1)],x1=0,x2=0,y1=1,y2=0),
                interpolate="monotone",
            ).encode(
                x=alt.X(f"{x_col}:N",sort=None,axis=alt.Axis(labelAngle=-38,labelLimit=130,title=x_label)),
                y=alt.Y(f"{y_col}:Q",axis=alt.Axis(title=y_label,grid=True),scale=alt.Scale(domain=[max(0,float(y_vals.min())*0.85),float(y_vals.max())*1.15])),
                tooltip=[alt.Tooltip(f"{x_col}:N",title=x_label),alt.Tooltip(f"{y_col}:Q",title=y_label,format=".1f")]
            )
            point_layer=alt.Chart(df).mark_circle(size=55,color="white",stroke=ACCENT,strokeWidth=2.5).encode(x=alt.X(f"{x_col}:N",sort=None),y=alt.Y(f"{y_col}:Q"))
            max_idx=int(np.argmax(y_vals)); min_idx=int(np.argmin(y_vals)); x_strs=df[x_col].astype(str).tolist()
            ann_rows=[{x_col:x_strs[i],y_col:float(y_vals[i]),"ann":l,"dy":d} for i,l,d in [(max_idx,f"Peak: {y_vals[max_idx]:.1f}",-22),(min_idx,f"Low: {y_vals[min_idx]:.1f}",20)]]
            ann_df=pd.DataFrame(ann_rows); ann_up=ann_df[ann_df["dy"]<0]; ann_dn=ann_df[ann_df["dy"]>0]
            annotations=alt.layer(
                alt.Chart(ann_up).mark_text(dy=-22,fontSize=10,fontWeight="bold",font=FONT,color=ACCENT).encode(x=alt.X(f"{x_col}:N",sort=None),y=alt.Y(f"{y_col}:Q"),text=alt.Text("ann:N")),
                alt.Chart(ann_dn).mark_text(dy=20,fontSize=10,fontWeight="bold",font=FONT,color=ACCENT).encode(x=alt.X(f"{x_col}:N",sort=None),y=alt.Y(f"{y_col}:Q"),text=alt.Text("ann:N")),
            )
            chart=(area+point_layer+annotations).properties(
                title=alt.TitleParams(text=title,subtitle="Trend over time"),
                width=max(560,len(df)*58),height=400).configure(**altair_config())
            out_path.write_bytes(safe_vegalite_to_png(chart, scale=2))

        elif chart_type == "multi_line":
            if len(df.columns)>3:
                id_col=df.columns[0]; val_cols=df.columns[1:]
                df=df.melt(id_vars=id_col,value_vars=val_cols,var_name="Series",value_name="Value")
                time_col=id_col; series_col="Series"; value_col="Value"
            elif len(df.columns)==3:
                time_col=df.columns[0]; series_col=df.columns[1]; value_col=df.columns[2]
            else:
                time_col=df.columns[0]; value_col=df.columns[1]; series_col=None
            df[value_col]=pd.to_numeric(df[value_col],errors='coerce'); df=df.dropna(subset=[value_col]); df[time_col]=df[time_col].astype(str); df=df.sort_values(time_col)
            series_list=df[series_col].unique().tolist() if series_col else []
            color_scale=alt.Scale(domain=series_list,range=PALETTE[:len(series_list)])
            base=alt.Chart(df).encode(
                x=alt.X(f"{time_col}:N",sort=None,axis=alt.Axis(labelAngle=-38,labelLimit=120,title="Month",grid=False,tickCount=len(df[time_col].unique()))),
                y=alt.Y(f"{value_col}:Q",axis=alt.Axis(title=y_label or value_col.replace("_"," ").title(),grid=True),scale=alt.Scale(domain=[0,float(df[value_col].max())*1.15])),
                color=alt.Color(f"{series_col}:N",scale=color_scale,legend=alt.Legend(title=None,orient="bottom",direction="horizontal",labelFontSize=12,labelFontWeight="bold",symbolSize=120,symbolStrokeWidth=3)) if series_col else alt.value(ACCENT),
            )
            lines=base.mark_line(strokeWidth=2.8,interpolate="monotone",point=False)
            points=base.mark_circle(size=55,opacity=0.9,stroke="white",strokeWidth=1.5)
            area=base.mark_area(interpolate="monotone",opacity=0.07)
            ann_rows=[]
            if series_col:
                for s in series_list:
                    s_df=df[df[series_col]==s]
                    if len(s_df)==0: continue
                    max_row=s_df.loc[s_df[value_col].idxmax()]
                    ann_rows.append({time_col:str(max_row[time_col]),series_col:s,value_col:float(max_row[value_col]),"ann":f"{s}: {int(max_row[value_col])}","dy":-16})
            if ann_rows:
                ann_df=pd.DataFrame(ann_rows)
                annotations=alt.Chart(ann_df).mark_text(fontSize=9.5,fontWeight="bold",font=FONT,dy=-16).encode(x=alt.X(f"{time_col}:N",sort=None),y=alt.Y(f"{value_col}:Q"),color=alt.Color(f"{series_col}:N",scale=color_scale,legend=None),text=alt.Text("ann:N"))
            else: annotations=alt.Chart(pd.DataFrame()).mark_text()
            chart=(area+lines+points+annotations).properties(
                title=alt.TitleParams(text=title,subtitle="Monthly trend comparison"),
                width=max(600,len(df[time_col].unique())*55),height=420).configure(**altair_config())
            out_path.write_bytes(safe_vegalite_to_png(chart, scale=2))

        elif chart_type == "pie":
            df[y_col]=pd.to_numeric(df[y_col],errors='coerce'); df=df.dropna(subset=[y_col])
            GRADE_COLORS={"EXCELLENT":"#27AE60","GOOD":"#82E0AA","MODERATE":"#F39C12","HIGH CONCERN":"#E74C3C","LIMITED":"#E74C3C","WELL_STOCKED":"#27AE60","ADEQUATE":"#82E0AA","FOOD_DESERT":"#C0392B","TOP PICK":"#1A9B5F","SOLID CHOICE":"#82E0AA","MODERATE PICK":"#F39C12","LIMITED APPEAL":"#E74C3C","AFFORDABLE":"#27AE60","AVERAGE":"#F39C12","BELOW_AVERAGE":"#E74C3C","NONE":"#BDC3C7"}
            fallback=["#2E86AB","#A23B72","#F18F01","#44BBA4","#E94F37","#6B4E71"]
            slice_colors=[GRADE_COLORS.get(l.upper().strip(),fallback[i%len(fallback)]) for i,l in enumerate(df[x_col].astype(str))]
            y_sum = float(df[y_col].sum())
            is_percentage_col = 90 <= y_sum <= 110
            if is_percentage_col:
                center_number = 51
                center_label  = "neighborhoods"
            else:
                center_number = int(round(y_sum))
                center_label  = "rated"
            fig=go.Figure()
            fig.add_trace(go.Pie(labels=df[x_col].astype(str),values=df[y_col],hole=0.50,
                marker=dict(colors=slice_colors,line=dict(color="white",width=3)),
                textinfo="label+percent",textfont=dict(size=13,family="Arial"),
                textposition="outside",hovertemplate="<b>%{label}</b><br>Value: %{value}<br>%{percent}<extra></extra>",
                pull=[0.04]*len(df),rotation=45,direction="clockwise",automargin=True))
            fig.update_layout(
                title=dict(text=f"<b>{title}</b>",font=dict(size=20,family="Arial",color="#1A1A2E"),x=0.05,xanchor="left"),
                paper_bgcolor="#FAFAFA",plot_bgcolor="#FAFAFA",font=dict(family="Arial",color="#1A1A2E"),
                showlegend=True,legend=dict(orientation="v",yanchor="middle",y=0.5,xanchor="left",x=1.02,font=dict(size=12)),
                margin=dict(l=60,r=160,t=110,b=60),
                annotations=[dict(
                    text=(
                        f"<b>{center_number}</b><br><span style='font-size:11px'>{center_label}</span>"
                        if is_percentage_col else
                        f"<b>{center_number} of 51</b><br><span style='font-size:11px'>rated</span>"
                    ),
                    x=0.5,y=0.5,font=dict(size=18,family="Arial",color="#1A1A2E"),
                    showarrow=False,xref="paper",yref="paper",xanchor="center",yanchor="middle")])
            fig.write_image(str(out_path),format="png",width=1100,height=700,scale=2)

        else:
            df[y_col]=pd.to_numeric(df[y_col],errors='coerce'); df=df.dropna(subset=[y_col]); df=df.sort_values(y_col,ascending=True)
            chart=alt.Chart(df).mark_bar(cornerRadiusTopRight=5,cornerRadiusBottomRight=5,height=28).encode(
                y=alt.Y(f"{x_col}:N",sort=None,axis=alt.Axis(title=None,labelLimit=220,labelFontWeight="bold")),
                x=alt.X(f"{y_col}:Q",scale=alt.Scale(domain=[0,float(df[y_col].max())*1.18])),
                color=alt.Color(f"{y_col}:Q",scale=alt.Scale(domain=[float(df[y_col].min()),float(df[y_col].max())],range=["#E74C3C","#F39C12","#27AE60"]),legend=None),
            ).properties(title=title,width=580,height=max(220,len(df)*50)).configure(**altair_config())
            out_path.write_bytes(safe_vegalite_to_png(chart, scale=2))

        # ── Post-render file check ────────────────────────────────────────────
        render_result = validator.validate(
            AgentType.GRAPHIC_CHART,
            {
                "sql":        sql,
                "chart_type": chart_type,
                "df":         df,
                "out_path":   str(out_path),
                "user_query": user_query,
                "attempt":    1,
            }
        )
        if not render_result.passed:
            render_result.print_summary()
        else:
            print(f"[Validator] Post-render file check PASSED")

        return str(out_path)

    finally:
        val_conn.close()

# ── Report-facing chart generators (called by router handle_report) ───────────
# These generate charts in the same style as the main generate_chart() function
# but are called directly with pre-fetched data dicts/DataFrames instead of SQL.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CHARTS_TEMP = OUTPUT_BASE / "reports" / "tmp_charts"
CHARTS_TEMP.mkdir(parents=True, exist_ok=True)

# Domain color tokens (matching report_agent design system)
_C_ACCENT      = "#2563EB"
_C_PRIMARY     = "#0F172A"
_C_SUBTEXT     = "#64748B"
_C_GREEN       = "#10B981"
_C_ORANGE      = "#F97316"
_C_TEAL        = "#14B8A6"
_C_RED         = "#EF4444"
_C_PURPLE      = "#8B5CF6"
_C_PINK        = "#EC4899"
_C_INDIGO      = "#6366F1"
_C_CYAN        = "#06B6D4"
_C_AMBER       = "#F59E0B"
_C_BG          = "#FFFFFF"

_DOMAIN_COLORS = {
    "Safety":       _C_GREEN,
    "Transit":      _C_ACCENT,
    "Housing":      _C_ORANGE,
    "Grocery":      _C_TEAL,
    "Healthcare":   _C_RED,
    "Schools":      _C_PURPLE,
    "Restaurants":  _C_PINK,
    "Universities": _C_INDIGO,
    "Bluebikes":    _C_CYAN,
}

_DOMAIN_ORDER = [
    "Safety", "Transit", "Housing", "Grocery",
    "Healthcare", "Schools", "Restaurants", "Universities", "Bluebikes",
]
_DOMAIN_COLS = {
    "Safety":       "SAFETY_SCORE",
    "Transit":      "TRANSIT_SCORE",
    "Housing":      "HOUSING_SCORE",
    "Grocery":      "GROCERY_SCORE",
    "Healthcare":   "HEALTHCARE_SCORE",
    "Schools":      "SCHOOL_SCORE",
    "Restaurants":  "RESTAURANT_SCORE",
    "Universities": "EDUCATION_SCORE",
    "Bluebikes":    "BIKESHARE_SCORE",
}

def _magazine_altair_config():
    return {
        "background": _C_BG, "font": "Helvetica",
        "title": {"fontSize": 18, "fontWeight": "bold", "color": _C_PRIMARY,
                  "anchor": "start", "offset": 14, "subtitleColor": _C_SUBTEXT,
                  "subtitleFontSize": 12, "subtitlePadding": 4},
        "axis": {"labelColor": "#334155", "labelFontSize": 11,
                 "titleColor": _C_SUBTEXT, "titleFontSize": 12,
                 "gridColor": "#F1F5F9", "gridOpacity": 1,
                 "domainColor": "#CBD5E1", "tickColor": "#CBD5E1",
                 "labelFont": "Helvetica", "titleFont": "Helvetica",
                 "labelPadding": 6, "titlePadding": 10},
        "legend": {"labelColor": _C_PRIMARY, "labelFontSize": 11,
                   "titleColor": _C_SUBTEXT, "labelFont": "Helvetica",
                   "orient": "bottom", "direction": "horizontal"},
        "view": {"stroke": "transparent", "fill": _C_BG},
    }


def generate_radar_chart(data: dict, neighborhood: str):
    """Radar chart for report — one spoke per domain."""
    try:
        scores = [float(data.get(_DOMAIN_COLS[d]) or 0) for d in _DOMAIN_ORDER]
        N      = len(_DOMAIN_ORDER)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        scores_plot = scores + scores[:1]
        domain_colors = [_DOMAIN_COLORS[d] for d in _DOMAIN_ORDER]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(_C_BG)
        ax.set_facecolor(_C_BG)

        for r in [25, 50, 75, 100]:
            ax.plot(angles, [r] * (N + 1), color="#E2E8F0", linewidth=0.6)

        ax.fill(angles, scores_plot, alpha=0.08, color=_C_ACCENT)
        ax.plot(angles, scores_plot, color=_C_ACCENT, linewidth=2.5)

        for i, (angle, score) in enumerate(zip(angles[:-1], scores)):
            color = domain_colors[i]
            ax.plot(angle, score, 'o', color=color, markersize=8, zorder=5,
                    markeredgecolor='white', markeredgewidth=2)
            label_r = score + 8 if score < 90 else score - 10
            ax.text(angle, label_r, f"{score:.0f}", ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.9, linewidth=0.8))

        ax.set_ylim(0, 115)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(_DOMAIN_ORDER, fontsize=10, color=_C_PRIMARY, fontweight='bold')
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color=_C_SUBTEXT)
        ax.grid(color="#E2E8F0", linewidth=0.5)
        ax.spines["polar"].set_color("#E2E8F0")
        ax.set_title(f"{neighborhood}", fontsize=16, color=_C_PRIMARY,
                     fontweight='bold', pad=24)

        ts  = datetime.now().strftime("%H%M%S")
        out = CHARTS_TEMP / f"radar_{neighborhood.lower().replace(' ', '_')}_{ts}.png"
        fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor=_C_BG)
        plt.close(fig)
        print(f"[Chart] Radar chart saved: {out.name}")
        return str(out)
    except Exception as e:
        print(f"[Chart] Radar chart failed: {e}")
        import traceback; traceback.print_exc()
        return None


def generate_bar_neighbors(data: dict, neighbor_df, neighborhood: str):
    """Horizontal bar chart comparing neighborhood to 5 nearest neighbors."""
    try:
        import pandas as pd
        if neighbor_df is None or (hasattr(neighbor_df, 'empty') and neighbor_df.empty):
            return None

        this_score = float(data.get("MASTER_SCORE") or 0)
        rows = [{"Neighborhood": neighborhood, "Score": this_score, "is_focus": True}]
        for _, row in neighbor_df.iterrows():
            rows.append({
                "Neighborhood": str(row["NEIGHBORHOOD_NAME"]).title(),
                "Score": float(row["MASTER_SCORE"] or 0),
                "is_focus": False,
            })
        df = pd.DataFrame(rows).sort_values("Score", ascending=True)

        color_scale = alt.Scale(domain=[True, False], range=[_C_ACCENT, "#CBD5E1"])
        bars = alt.Chart(df).mark_bar(
            cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=28,
        ).encode(
            y=alt.Y("Neighborhood:N", sort=None,
                    axis=alt.Axis(title=None, labelLimit=200,
                                  labelFontWeight="bold", labelFontSize=12)),
            x=alt.X("Score:Q",
                    axis=alt.Axis(title="Master Score", grid=True, tickCount=5),
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("is_focus:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("Neighborhood:N"),
                     alt.Tooltip("Score:Q", format=".1f")],
        )
        text = alt.Chart(df).mark_text(
            align="left", dx=6, fontSize=12, fontWeight="bold", color=_C_PRIMARY,
        ).encode(
            y=alt.Y("Neighborhood:N", sort=None),
            x=alt.X("Score:Q"),
            text=alt.Text("Score:Q", format=".1f"),
        )
        chart = (bars + text).properties(
            title=alt.TitleParams(
                text=f"How {neighborhood} compares",
                subtitle="Master score vs nearby neighborhoods  ·  Blue = this neighborhood",
            ),
            width=500, height=max(180, len(df) * 42),
        ).configure(**_magazine_altair_config())

        ts  = datetime.now().strftime("%H%M%S")
        out = CHARTS_TEMP / f"bar_neighbors_{neighborhood.lower().replace(' ', '_')}_{ts}.png"
        out.write_bytes(safe_vegalite_to_png(chart, scale=2))
        print(f"[Chart] Bar neighbors chart saved: {out.name}")
        return str(out)
    except Exception as e:
        print(f"[Chart] Bar neighbors chart failed: {e}")
        import traceback; traceback.print_exc()
        return None


def generate_grouped_bar(data: dict, neighborhood: str):
    """Horizontal grouped bar — all 9 domain scores for the report."""
    try:
        import pandas as pd
        rows = []
        for domain in _DOMAIN_ORDER:
            score = data.get(_DOMAIN_COLS[domain])
            if score is not None:
                rows.append({"Domain": domain, "Score": float(score)})
        if not rows:
            return None

        df = pd.DataFrame(rows).sort_values("Score", ascending=True)
        color_scale = alt.Scale(
            domain=list(_DOMAIN_COLORS.keys()),
            range=list(_DOMAIN_COLORS.values()),
        )
        bars = alt.Chart(df).mark_bar(
            cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=26,
        ).encode(
            y=alt.Y("Domain:N", sort=None,
                    axis=alt.Axis(title=None, labelFontSize=11,
                                  labelFontWeight="bold", labelLimit=150)),
            x=alt.X("Score:Q",
                    axis=alt.Axis(title="Score (0–100)", grid=True),
                    scale=alt.Scale(domain=[0, 110])),
            color=alt.Color("Domain:N", scale=color_scale, legend=None),
            tooltip=[alt.Tooltip("Domain:N"),
                     alt.Tooltip("Score:Q", format=".1f")],
        )
        text = alt.Chart(df).mark_text(
            align="left", dx=6, fontSize=11, fontWeight="bold", color=_C_PRIMARY,
        ).encode(
            y=alt.Y("Domain:N", sort=None),
            x=alt.X("Score:Q"),
            text=alt.Text("Score:Q", format=".0f"),
        )
        chart = (bars + text).properties(
            title=alt.TitleParams(
                text=f"{neighborhood} — domain scorecard",
                subtitle="Each domain scored 0–100  ·  Higher is better",
            ),
            width=480, height=max(220, len(df) * 36),
        ).configure(**_magazine_altair_config())

        ts  = datetime.now().strftime("%H%M%S")
        out = CHARTS_TEMP / f"grouped_bar_{neighborhood.lower().replace(' ', '_')}_{ts}.png"
        out.write_bytes(safe_vegalite_to_png(chart, scale=2))
        print(f"[Chart] Grouped bar chart saved: {out.name}")
        return str(out)
    except Exception as e:
        print(f"[Chart] Grouped bar chart failed: {e}")
        import traceback; traceback.print_exc()
        return None


def generate_crime_trend(crime_df, neighborhood: str, forecast_df=None):
    """Crime trend area chart + SARIMAX forecast overlay."""
    try:
        import pandas as pd
        if crime_df is None or (hasattr(crime_df, 'empty') and crime_df.empty) \
                or len(crime_df) < 3:
            return None

        crime_df = crime_df.copy()
        crime_df["YEAR_MONTH"]  = crime_df["YEAR_MONTH"].astype(str)
        crime_df["CRIME_COUNT"] = pd.to_numeric(crime_df["CRIME_COUNT"], errors="coerce")
        crime_df = crime_df.dropna(subset=["CRIME_COUNT"])
        if crime_df.empty:
            return None

        y_vals = crime_df["CRIME_COUNT"].values
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        crime_df["is_max"] = crime_df["CRIME_COUNT"] == y_max
        crime_df["is_min"] = crime_df["CRIME_COUNT"] == y_min
        crime_df["TYPE"]   = "Actual"

        has_forecast = (forecast_df is not None
                        and not (hasattr(forecast_df, 'empty') and forecast_df.empty))
        combined_df  = crime_df.copy()

        if has_forecast:
            fc = forecast_df.copy()
            fc["YEAR_MONTH"]  = fc["FORECAST_MONTH"].astype(str)
            fc["CRIME_COUNT"] = pd.to_numeric(fc["FORECASTED_COUNT"], errors="coerce")
            fc["LOWER_CI"]    = pd.to_numeric(fc["LOWER_CI"],          errors="coerce")
            fc["UPPER_CI"]    = pd.to_numeric(fc["UPPER_CI"],          errors="coerce")
            fc["TYPE"]        = "Forecast"
            fc["is_max"]      = False
            fc["is_min"]      = False
            last_actual       = crime_df.iloc[-1:].copy()
            last_actual["TYPE"]     = "Forecast"
            last_actual["LOWER_CI"] = last_actual["CRIME_COUNT"]
            last_actual["UPPER_CI"] = last_actual["CRIME_COUNT"]
            fc_with_bridge = pd.concat([last_actual, fc], ignore_index=True)
            combined_df    = pd.concat(
                [crime_df,
                 fc[["YEAR_MONTH","CRIME_COUNT","LOWER_CI","UPPER_CI",
                      "TYPE","is_max","is_min"]]],
                ignore_index=True,
            )
            # Sort chronologically so x-axis is left=oldest, right=newest
            combined_df    = combined_df.sort_values("YEAR_MONTH").reset_index(drop=True)
            fc_with_bridge = fc_with_bridge.sort_values("YEAR_MONTH").reset_index(drop=True)
            y_max = max(y_max, float(fc["UPPER_CI"].max()))

        all_months  = sorted(combined_df["YEAR_MONTH"].tolist())
        n_points    = len(all_months)
        tick_every  = 1 if n_points <= 12 else (2 if n_points <= 24 else 3)
        show_months = [m for i, m in enumerate(all_months) if i % tick_every == 0]

        actual_df = combined_df[combined_df["TYPE"] == "Actual"]

        area = alt.Chart(actual_df).mark_area(
            line={"color": _C_ACCENT, "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="rgba(37,99,235,0.20)", offset=0),
                       alt.GradientStop(color="rgba(37,99,235,0.02)", offset=1)],
                x1=0, x2=0, y1=1, y2=0,
            ),
            interpolate="monotone",
        ).encode(
            x=alt.X("YEAR_MONTH:O", sort=all_months,
                    axis=alt.Axis(labelAngle=-40, labelLimit=120,
                                  title="Month", values=show_months)),
            y=alt.Y("CRIME_COUNT:Q",
                    axis=alt.Axis(title="Monthly incidents", grid=True),
                    scale=alt.Scale(domain=[
                        max(0, float(actual_df["CRIME_COUNT"].min()) * 0.80),
                        y_max * 1.20,
                    ])),
            tooltip=[alt.Tooltip("YEAR_MONTH:O", title="Month"),
                     alt.Tooltip("CRIME_COUNT:Q", title="Incidents", format=".0f")],
        )
        points    = alt.Chart(actual_df).mark_circle(
            size=35, color="white", stroke=_C_ACCENT, strokeWidth=1.5,
        ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"))
        max_point = alt.Chart(actual_df[actual_df["is_max"]]).mark_circle(
            size=80, color=_C_RED, stroke="white", strokeWidth=2,
        ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"))
        min_point = alt.Chart(actual_df[actual_df["is_min"]]).mark_circle(
            size=80, color=_C_GREEN, stroke="white", strokeWidth=2,
        ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"))
        max_label = alt.Chart(actual_df[actual_df["is_max"]].head(1)).mark_text(
            dy=-14, fontSize=11, fontWeight="bold", color=_C_RED,
        ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"),
                 text=alt.Text("CRIME_COUNT:Q", format=".0f"))
        min_label = alt.Chart(actual_df[actual_df["is_min"]].head(1)).mark_text(
            dy=14, fontSize=11, fontWeight="bold", color=_C_GREEN,
        ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"),
                 text=alt.Text("CRIME_COUNT:Q", format=".0f"))

        layers = [area, points, max_point, min_point, max_label, min_label]

        if has_forecast:
            ci_band = alt.Chart(fc_with_bridge).mark_area(
                opacity=0.15, color=_C_ORANGE, interpolate="monotone",
            ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months),
                     y=alt.Y("LOWER_CI:Q"), y2="UPPER_CI:Q")
            fc_line = alt.Chart(fc_with_bridge).mark_line(
                strokeDash=[6, 3], strokeWidth=2.5, color=_C_ORANGE,
                interpolate="monotone",
            ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"))
            fc_pts  = alt.Chart(fc[["YEAR_MONTH","CRIME_COUNT"]]).mark_circle(
                size=50, color=_C_ORANGE, stroke="white", strokeWidth=2,
            ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"))
            fc_lbl  = alt.Chart(fc[["YEAR_MONTH","CRIME_COUNT"]]).mark_text(
                dy=-12, fontSize=10, fontWeight="bold", color=_C_ORANGE,
            ).encode(x=alt.X("YEAR_MONTH:O", sort=all_months), y=alt.Y("CRIME_COUNT:Q"),
                     text=alt.Text("CRIME_COUNT:Q", format=".0f"))
            layers = [ci_band] + layers + [fc_line, fc_pts, fc_lbl]
            mape_val   = forecast_df["TRAIN_MAPE"].iloc[0] \
                if "TRAIN_MAPE" in forecast_df.columns else None
            mape_str   = f"  ·  Model MAPE: {mape_val:.1f}%" if mape_val is not None else ""
            subtitle   = f"Blue = actual  ·  Orange dashed = SARIMAX forecast{mape_str}"
        else:
            subtitle = "Red = peak  ·  Green = lowest"

        chart = alt.layer(*layers).properties(
            title=alt.TitleParams(
                text=f"{neighborhood} — crime trend & forecast",
                subtitle=subtitle,
            ),
            width=max(500, len(combined_df) * 18),
            height=260,
        ).configure(**_magazine_altair_config())

        ts  = datetime.now().strftime("%H%M%S")
        out = CHARTS_TEMP / f"crime_trend_{neighborhood.lower().replace(' ', '_')}_{ts}.png"
        out.write_bytes(safe_vegalite_to_png(chart, scale=2))
        print(f"[Chart] Crime trend chart saved: {out.name}")
        return str(out)
    except Exception as e:
        print(f"[Chart] Crime trend chart failed: {e}")
        import traceback; traceback.print_exc()
        return None

# ── Housing type lookup ────────────────────────────────────────────────────────
NEIGHBORHOOD_HOUSING_TYPES = {
    "DORCHESTER": "wooden triple-decker houses — 3 stories tall, each floor has a flat covered porch running the full width, wood siding painted in faded yellows/greens/blues, detached with small front yards and chain-link fences",
    "ROXBURY": "wooden triple-decker houses mixed with 2-3 story attached brick rowhouses — triple-deckers have flat covered porches on each of 3 floors",
    "JAMAICA PLAIN": "wooden triple-decker houses 3 stories tall with flat covered porches, some 2-3 story brick rowhouses on busier streets",
    "ALLSTON": "wooden triple-decker houses 3 stories tall with flat porches on each level, wood siding in faded colors",
    "BRIGHTON": "wooden triple-decker houses 3 stories tall with flat covered porches",
    "EAST BOSTON": "wooden triple-decker houses 3 stories tall with flat covered porches, wood siding in pale yellows and blues",
    "MATTAPAN": "wooden triple-decker houses 3 stories tall with flat covered porches on each floor",
    "HYDE PARK": "wooden triple-decker houses 3 stories tall with flat covered porches, some single-family detached wooden houses",
    "ROSLINDALE": "wooden triple-decker houses 3 stories tall with flat covered porches",
    "SOMERVILLE": "wooden triple-decker houses 3 stories tall with flat covered porches, painted in greens/yellows/blues",
    "MEDFORD": "wooden triple-decker houses 3 stories tall — each floor has a flat covered porch running the full width of the house, wood siding painted in faded greens/yellows/whites, small front yards with chain-link or wooden fences, narrow driveways between houses. Wide tree-lined residential streets.",
    "MALDEN": "wooden triple-decker houses 3 stories tall with flat covered porches, wood siding in varied colors",
    "EVERETT": "wooden triple-decker houses 3 stories tall with flat covered porches",
    "CHELSEA": "wooden triple-decker houses mixed with 2-3 story attached brick buildings",
    "REVERE": "wooden triple-decker houses 3 stories tall with flat covered porches",
    "BEACON HILL": "attached 4-story Federal-style red brick rowhouses sharing walls, white marble stoops of 6-8 steps, black iron railings, black shutters, narrow cobblestone streets",
    "BACK BAY": "attached 4-5 story brownstone rowhouses with distinctive bow bay windows on every floor, high stoops of 10-12 steps with ornate iron railings",
    "SOUTH END": "attached 4-story Victorian brownstone and red brick rowhouses with bow bay windows, high stoops of 8-10 steps",
    "CHARLESTOWN": "attached 3-4 story Federal-style red brick rowhouses, modest stoops, flat rooflines",
    "NORTH END": "dense 3-4 story attached red brick buildings along very narrow streets",
    "DOWNTOWN": "mix of historic 4-story brick commercial buildings and modern glass towers",
}

def get_expected_housing_type(neighborhood: str) -> str:
    return NEIGHBORHOOD_HOUSING_TYPES.get(
        neighborhood.upper(),
        "Boston-area urban housing — brick rowhouses, brownstones, or wooden triple-deckers with front stoops or flat porches"
    )

# ── Landmark Visual Translator ─────────────────────────────────────────────────
def translate_landmarks_to_visual_descriptions(landmarks, neighborhood, city):
    landmark_list = ", ".join(landmarks)
    housing_type  = get_expected_housing_type(neighborhood)
    prompt = (
        f"You are directing a film crew to shoot a street-level exterior scene "
        f"of the {neighborhood} neighborhood in {city}, Massachusetts.\n\n"
        f"Real landmarks nearby: {landmark_list}\n\n"
        f"CRITICAL — The correct housing type for {neighborhood} is:\n"
        f"{housing_type}\n\n"
        f"Write ONE paragraph (max 220 words) describing ONLY what the camera "
        f"would see from the sidewalk. The housing type described above MUST "
        f"be the dominant building type in your description.\n\n"
        f"STRICT RULES:\n"
        f"1. EXTERIOR ONLY — no interiors.\n"
        f"2. Housing type MUST match the description above exactly.\n"
        f"3. NO proper nouns, NO brand names, NO team names.\n"
        f"4. NO text, NO signs, NO readable words anywhere.\n"
        f"5. Specific: colors, materials, stories, porch/stoop details.\n"
        f"6. Describe the street: width, surface, pedestrians, cars, trees.\n"
        f"7. This must look like {neighborhood}, Massachusetts — not generic Boston."
    )
    conn = get_snowflake_conn()
    try:
        cur = conn.cursor()
        safe_prompt = prompt.replace("'", "\\'")
        cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{safe_prompt}')")
        return cur.fetchone()[0].strip()
    except Exception as e:
        return f"exterior street view in {neighborhood} {city}"
    finally:
        conn.close()

def get_neighborhood_landmarks(neighborhood, city):
    key = neighborhood.upper()
    if key in NEIGHBORHOOD_OVERRIDES:
        print(f"[Image] Using hardcoded override for '{neighborhood}'.")
        return NEIGHBORHOOD_OVERRIDES[key]
    cached = get_cached_landmarks(neighborhood)
    if cached: return cached
    print(f"[Image] Searching Tavily for landmarks in {neighborhood}, {city}...")
    try:
        results = tavily_client.search(
            query=f"famous landmarks iconic places architecture streets {neighborhood} {city} Massachusetts",
            search_depth="basic", max_results=3
        )
        snippets=[r.get("content","")[:300] for r in results.get("results",[]) if r.get("content")]
        if not snippets: return {}
        combined=" ".join(snippets)
        extract_prompt=(f"From this text about {neighborhood} in {city}, Massachusetts, extract a comma-separated list of 3-5 specific landmark names, iconic streets, or famous places. Return ONLY the names separated by commas. No explanation.\n\nText: {combined[:800]}")
        conn=get_snowflake_conn()
        try:
            cur=conn.cursor()
            cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{extract_prompt.replace(chr(39), chr(92)+chr(39))}')")
            result=cur.fetchone()[0].strip()
            landmarks=[l.strip() for l in result.split(",") if l.strip()]
        finally:
            conn.close()
        if not landmarks: return {}
        visual_desc=translate_landmarks_to_visual_descriptions(landmarks,neighborhood,city)
        data={"names":landmarks,"visual":visual_desc}
        cache_landmarks(neighborhood, data)
        return data
    except Exception as e:
        print(f"[Image] Landmark search failed: {e}")
        return {}

def get_neighborhood_transit_lines(neighborhood: str) -> dict:
    try:
        sql = f"""
            SELECT DISTINCT ROUTE_NAME, ROUTE_TYPE
            FROM MARTS.MRT_BOSTON_TRANSIT_ROUTES_BY_NEIGHBORHOOD
            WHERE UPPER(NEIGHBORHOOD_NAME) = '{neighborhood.upper()}'
            ORDER BY ROUTE_TYPE, ROUTE_NAME
        """
        df = run_query(sql)
        if df.empty:
            return {}
        rapid_lines=[]; commuter_lines=[]; bus_count=0
        for _, row in df.iterrows():
            name  = str(row.get("ROUTE_NAME","") or "").strip()
            rtype = str(row.get("ROUTE_TYPE","") or "").strip().lower()
            if not name: continue
            if "commuter" in rtype or "commuter" in name.lower():
                commuter_lines.append(name)
            elif "bus" in rtype or name.startswith(tuple("0123456789")):
                bus_count += 1
            elif any(c in name.lower() for c in ["green","red","orange","blue","silver"]):
                rapid_lines.append(name)
            else:
                rapid_lines.append(name)
        return {
            "rapid_lines":    list(dict.fromkeys(rapid_lines)),
            "commuter_lines": list(dict.fromkeys(commuter_lines)),
            "has_bus":        bus_count > 0,
            "bus_count":      bus_count,
        }
    except Exception as e:
        print(f"[Image] Transit lines lookup failed (non-critical): {e}")
        return {}

LINE_VISUALS = {
    "green line": (
        "a Green Line MBTA light rail stop — a low street-level platform "
        "with green painted steel canopy, a green articulated trolley car "
        "on the track at street level, overhead wires above the track"
    ),
    "red line": (
        "a Red Line MBTA subway entrance on the sidewalk — a brick and "
        "concrete headhouse structure with a metal canopy over a wide "
        "staircase descending underground into darkness below street level. "
        "The station is entirely underground — only the entrance kiosk is "
        "visible at street level, no trains visible above ground, "
        "no elevated tracks, no surface platforms. "
        "Red accent color on the entrance canopy and signage panels."
    ),
    "orange line": (
        "an Orange Line MBTA subway entrance on the sidewalk — a concrete "
        "and steel headhouse with orange accent elements over a staircase "
        "going down into an underground station below street level. "
        "The station is fully underground — only the entrance structure "
        "is visible at street level, no trains or tracks visible above ground. "
        "Commuters walking down the stairs into the underground entrance."
    ),
    "blue line": (
        "a Blue Line MBTA subway entrance — a metal canopy over concrete "
        "stairs descending underground, blue accent color on the entrance "
        "structure, fully below-grade station with only the street-level "
        "entrance headhouse visible"
    ),
    "silver line": (
        "a Silver Line MBTA bus rapid transit stop — modern enclosed shelter "
        "with silver-grey design, BRT platform at street level"
    ),
}

def build_transit_constraints(info: dict, transit_lines: dict = None) -> tuple:
    rapid  = int(info.get("RAPID_TRANSIT_ROUTES") or 0)
    bus    = int(info.get("BUS_ROUTES")            or 0)
    grade  = str(info.get("TRANSIT_GRADE")         or "").upper()
    rapid_lines    = (transit_lines or {}).get("rapid_lines",    [])
    commuter_lines = (transit_lines or {}).get("commuter_lines", [])
    has_bus        = bus > 0 or (transit_lines or {}).get("has_bus", False)
    present=[]; absent=[]
    if rapid_lines:
        for line in rapid_lines[:2]:
            matched = next((v for k,v in LINE_VISUALS.items() if k in line.lower()), None)
            if matched:
                present.append(matched)
                print(f"[Image] Transit: using line-specific visual for '{line}'")
            else:
                present.append(f"an MBTA {line} station entrance headhouse at street level")
    elif rapid > 0:
        present.append(
            "an MBTA subway station entrance — brick headhouse with metal "
            "canopy over stairs going underground, green painted steel columns"
        )
    else:
        absent.extend(["subway station","tram","light rail","streetcar tracks",
                        "overhead electric wires","rail tracks on street"])
    if commuter_lines:
        present.append(
            "a commuter rail platform with silver MBTA bi-level coaches "
            "visible on the track beside a low-level platform"
        )
    if has_bus:
        present.append(
            "an MBTA bus stop with a green metal pole and small shelter, "
            "a white MBTA bus with blue stripe visible on the street"
        )
    else:
        absent.append("bus stops")
    if grade in ("LIMITED","MODERATE") or (rapid==0 and bus<=2):
        absent.extend(["elevated train tracks","rail infrastructure","train platforms"])
    present_str = ". ".join(present) if present else ""
    absent_str  = ("Strictly exclude from the scene: "+", ".join(absent)+"." if absent else "")
    return present_str, absent_str

def generate_single_image(neighborhood,city,perspective,visual_str,transit_present,transit_absent,context_str,index,housing_type=""):
    label=perspective["label"]; description=perspective["description"]; focus=perspective["prompt_focus"]
    transit_fill=""
    if transit_present: transit_fill+=f"Transit present: {transit_present}. "
    if transit_absent:  transit_fill+=f"{transit_absent} "
    focus_filled=focus.replace("{visual}",visual_str).replace("{transit}",transit_fill)
    housing_prefix=""
    if housing_type:
        housing_prefix=(
            f"SCENE REQUIREMENT — The dominant building type in this image MUST be: "
            f"{housing_type}. "
            f"Do NOT show rowhouses, brownstones, or any other building type. "
            f"ONLY show the specified housing type above. "
        )
    prompt=(
        f"{housing_prefix}"
        f"Photorealistic cinematic street-level photograph in the {neighborhood} neighborhood of {city}, Massachusetts. "
        f"Scene: {description}. {focus_filled} "
        f"CRITICAL RULES: (1) Show ONLY building exteriors and street — never any interior. "
        f"(2) No fantasy architecture, no castles, no ornate turrets. "
        f"(3) Must look like a REAL Boston-area neighborhood — not European, not South Asian market, not Indian bazaar, not Latin American market, not generic American suburb. "
        f"(4) ABSOLUTELY ZERO readable text anywhere in the entire image — "
        f"no store names, no street signs, no STOP signs, no transit signs, "
        f"no route numbers, no station names, no posters, no license plates with letters. "
        f"Every sign, awning, and shelter must be blank or have text facing away. "
        f"(5) HOUSING TYPE IS MANDATORY: {housing_type if housing_type else 'Boston-area urban housing'}. "
        f"Strictly no substitutions — do not replace with Victorian rowhouses, brownstones, or any other type. "
        f"New England urban character: iron fire escapes, concrete sidewalks, utility poles, parked cars, diverse pedestrians. "
        f"Style: 8K professional urban street photography, sharp detail, realistic proportions, authentic to the real {neighborhood} neighborhood in {city}, Massachusetts."
    )
    MAX_PROMPT=3950
    if len(prompt)>MAX_PROMPT:
        trimmed=focus_filled
        while len(prompt)>MAX_PROMPT and len(trimmed)>100:
            trimmed=trimmed[:len(trimmed)-150].rsplit(" ",1)[0]
            prompt=(
                f"{housing_prefix}"
                f"Photorealistic street-level photograph of {neighborhood}, {city}, Massachusetts. "
                f"Scene: {description}. {trimmed} "
                f"CRITICAL: (1) Exterior only. (2) Zero readable text anywhere. "
                f"(3) HOUSING TYPE MANDATORY: {housing_type if housing_type else 'Boston-area urban housing'}. No substitutions. "
                f"(4) Authentic New England — iron fire escapes, concrete sidewalks, utility poles. "
                f"Style: 8K urban street photography, authentic to {neighborhood}, {city}, Massachusetts."
            )
        print(f"[Image {index}/4] Prompt trimmed to {len(prompt)} chars to fit DALL-E limit")
    print(f"\n[Image {index}/4] Perspective: {description}")
    response=openai_client.images.generate(model="dall-e-3",prompt=prompt,size="1792x1024",quality="hd",n=1)
    image_url=response.data[0].url; img_response=requests.get(image_url); img_response.raise_for_status()
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"); safe_name=re.sub(r'[^a-zA-Z0-9_]','_',neighborhood.lower())
    out_path=IMAGES_DIR/f"{safe_name}_{index}_{label}_{timestamp}.png"; out_path.write_bytes(img_response.content)
    print(f"[Image {index}/4] Saved: {out_path.name}")
    return str(out_path)

def _stable_image_path(neighborhood: str, perspective_label: str) -> Path:
    """
    Returns a stable (non-timestamped) path for a cached image.
    e.g. IMAGES_DIR / "fenway_landmark.png"
    """
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', neighborhood.lower())
    return IMAGES_DIR / f"{safe_name}_{perspective_label}.png"

def get_cached_images(neighborhood: str) -> list:
    """
    Check if all 4 perspective images already exist on disk for this neighborhood.
    Returns list of paths if ALL 4 exist, empty list if any are missing.
    
    Checks both stable names (fenway_landmark.png) and timestamped names
    (fenway_1_landmark_20260403_*.png) from previous runs.
    """
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', neighborhood.lower())
    
    # First check stable cached names
    stable_paths = []
    for persp in IMAGE_PERSPECTIVES:
        p = _stable_image_path(neighborhood, persp["label"])
        if p.exists() and p.stat().st_size > 50_000:  # >50KB = valid DALL-E image
            stable_paths.append(str(p))
    
    if len(stable_paths) == 4:
        print(f"[Image] ✅ Cache hit — all 4 images found for '{neighborhood}' (stable names)")
        return stable_paths
    
    # Fallback: check for timestamped files from previous runs
    # Pattern: fenway_1_landmark_*.png, fenway_2_residential_*.png, etc.
    found = {}
    for f in IMAGES_DIR.glob(f"{safe_name}_*_*.png"):
        for persp in IMAGE_PERSPECTIVES:
            if f"_{persp['label']}_" in f.name and f.stat().st_size > 50_000:
                if persp["label"] not in found:
                    found[persp["label"]] = str(f)
    
    if len(found) == 4:
        print(f"[Image] ✅ Cache hit — all 4 images found for '{neighborhood}' (timestamped)")
        return [found[p["label"]] for p in IMAGE_PERSPECTIVES]
    
    missing = [p["label"] for p in IMAGE_PERSPECTIVES if p["label"] not in found]
    print(f"[Image] Cache miss for '{neighborhood}' — missing: {missing}")
    return []

def generate_neighborhood_images(neighborhood):
    """
    Generate 4 DALL-E perspective images for a neighborhood.
    
    CACHING: Checks disk first. Only calls DALL-E for missing perspectives.
    Saves with stable filenames so subsequent runs find them instantly.
    Cost: $0.00 on cache hit, $0.08 per missing image on cache miss.
    """
    # ── CHECK CACHE FIRST ─────────────────────────────────────────────────────
    cached = get_cached_images(neighborhood)
    if cached:
        print(f"[Image] Using {len(cached)} cached images for '{neighborhood}' — $0.00 DALL-E cost")
        # Skip validation on cached images — they were validated when first generated
        return cached

    # ── CACHE MISS — generate from DALL-E ─────────────────────────────────────
    print(f"\n[Image] Fetching neighborhood context for '{neighborhood}'...")
    info = get_neighborhood_narrative(neighborhood)
    city = "Boston"
    if info:
        city = str(info.get("CITY", "Boston")).title()

    landmark_data = get_neighborhood_landmarks(neighborhood, city)
    visual_str = ""
    housing_type = get_expected_housing_type(neighborhood)

    if landmark_data:
        names = landmark_data.get("names", [])
        visual = landmark_data.get("visual", "")
        if names:
            print(f"[Image] Landmarks: {names}")
        if visual:
            visual_str = f"Street-level exterior features: {visual} Show only exteriors — never interiors of any structure."

    visual_str += (
        f" CRITICAL HOUSING TYPE for {neighborhood}: {housing_type}. "
        f"Every building shown MUST match this housing type exactly."
    )

    transit_present, transit_absent = ("", "")
    if info:
        transit_lines = get_neighborhood_transit_lines(neighborhood)
        if transit_lines.get("rapid_lines"):
            print(f"[Image] Rapid transit lines: {transit_lines['rapid_lines']}")
        if transit_lines.get("commuter_lines"):
            print(f"[Image] Commuter lines: {transit_lines['commuter_lines']}")
        transit_present, transit_absent = build_transit_constraints(info, transit_lines)

    context_str = ""
    if info:
        safety = str(info.get("SAFETY_DESCRIPTION", "") or "")[:100]
        food = str(info.get("RESTAURANT_DESCRIPTION", "") or "")[:100]
        context_str = " ".join([p for p in [safety, food] if p])[:200]

    print(f"\n[Image] Generating 4 images for '{neighborhood}'...\n")
    print(f"[Image] Housing type: {housing_type[:80]}...")

    saved_paths = []
    for i, perspective in enumerate(IMAGE_PERSPECTIVES, start=1):
        try:
            # Generate the image via DALL-E
            path = generate_single_image(
                neighborhood, city, perspective, visual_str,
                transit_present, transit_absent, context_str, i, housing_type
            )
            saved_paths.append(path)

            # Copy to stable cached name for future lookups
            stable = _stable_image_path(neighborhood, perspective["label"])
            if path and Path(path).exists():
                import shutil
                shutil.copy2(path, stable)
                print(f"[Image] Cached: {stable.name}")

        except Exception as e:
            print(f"[Image {i}/4] Failed: {e}")

# ── Validate all generated images ────────────────────────────────────────
    try:
        validate_images(neighborhood, city, saved_paths, info)
    except Exception as e:
        print(f"[Image] Validation skipped: {e}")
    
    return saved_paths

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 graphic_agent.py \"your query here\"")
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    print(f"\n{'='*60}")
    print(f"NeighbourWise Graphic Agent")
    print(f"Query: {user_query}")
    print(f"{'='*60}\n")

    print("[Router] Classifying intent via Snowflake Cortex...")
    plan = classify_intent(user_query)
    print(f"[Router] Intent     : {plan.get('intent')}")
    print(f"[Router] Chart type : {plan.get('chart_type')}")
    print(f"[Router] Reason     : {plan.get('reasoning')}\n")

    intent = plan.get("intent")

    if intent == "chart":
        out = generate_chart(plan, user_query)
        if out:
            print(f"\n✅ Chart saved to: {out}")
        else:
            print("\n❌ Chart generation failed.")

    elif intent == "image":
        neighborhood = plan.get("neighborhood", "").strip()
        if not neighborhood:
            print("[Image] Could not extract neighborhood name.")
            sys.exit(1)
        print(f"[Router] Neighborhood: {neighborhood}\n")
        saved = generate_neighborhood_images(neighborhood)
        if saved:
            print(f"\n✅ {len(saved)}/4 images generated for '{neighborhood}':")
            for p in saved: print(f"   → {p}")
        else:
            print("\n❌ All image generations failed.")
    else:
        print(f"❌ Could not classify intent: {plan.get('reasoning')}")

if __name__ == "__main__":
    main()