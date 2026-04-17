"""
report_agent.py — NeighbourWise AI (v3 — Assembly Edition)
═══════════════════════════════════════════════════════════
ARCHITECTURE CHANGE from v2.1:
  - Charts are NO LONGER generated inside this file
  - All 4 charts must be passed in via precomputed["chart_paths"]
  - This file is now responsible for:
      1. Fetching domain data + crime data + RAG (when run standalone)
      2. Generating Cortex text narratives (exec summary, recommendation)
      3. Assembling the PDF from pre-generated charts + fetched data

TWO MODES:
  Standalone:  python3 report_agent.py "Fenway"
               → fetches everything itself, calls GraphicAgent internally
               → backward compatible, nothing breaks

  Via Router:  generate_report(neighborhood, precomputed=precomputed_dict)
               → uses whatever the router already computed
               → no redundant Snowflake calls, no redundant chart generation

precomputed dict schema:
{
    "data":            dict,        # from fetch_domain_data()
    "neighbor_df":     DataFrame,
    "crime_df":        DataFrame,
    "forecast_df":     DataFrame,
    "crime_narrative": dict,
    "rag_narrative":   str,         # already summarized
    "narratives":      dict,        # exec summary + recommendation
    "chart_paths": {
        "chart_radar":          str,  # file path from GraphicAgent
        "chart_bar_neighbors":  str,
        "chart_grouped_bar":    str,
        "chart_crime_trend":    str,
    }
}
Any missing key → falls back to generating it internally.

Dependencies:
    pip install reportlab snowflake-connector-python python-dotenv
"""

import os
import sys
import re
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, HRFlowable, KeepTogether,
    Flowable, CondPageBreak,
)

# ── Universal Validator (replaces validator_report_agent) ─────────────────────
from universal_validator import UniversalValidator, AgentType

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
SNOWFLAKE_ACCOUNT   = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER      = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD  = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE  = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_ROLE      = os.getenv("SNOWFLAKE_ROLE")

OUTPUT_BASE = Path(__file__).parent / "outputs" / "reports"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────────
C_PRIMARY      = "#0F172A"
C_SECONDARY    = "#334155"
C_ACCENT       = "#2563EB"
C_ACCENT_LIGHT = "#DBEAFE"
C_ACCENT_DARK  = "#1E40AF"
C_SUBTEXT      = "#64748B"
C_GRID         = "#F1F5F9"
C_BG           = "#FFFFFF"
C_GREEN        = "#10B981"
C_GREEN_LIGHT  = "#D1FAE5"
C_AMBER        = "#F59E0B"
C_AMBER_LIGHT  = "#FEF3C7"
C_RED          = "#EF4444"
C_RED_LIGHT    = "#FEE2E2"
C_PURPLE       = "#8B5CF6"
C_TEAL         = "#14B8A6"
C_ORANGE       = "#F97316"
C_PINK         = "#EC4899"
C_INDIGO       = "#6366F1"
C_CYAN         = "#06B6D4"

DOMAIN_COLORS = {
    "Safety":       C_GREEN,
    "Transit":      C_ACCENT,
    "Housing":      C_ORANGE,
    "Grocery":      C_TEAL,
    "Healthcare":   C_RED,
    "Schools":      C_PURPLE,
    "Restaurants":  C_PINK,
    "Universities": C_INDIGO,
    "Bluebikes":    C_CYAN,
}

DOMAIN_ORDER = [
    "Safety", "Transit", "Housing", "Grocery",
    "Healthcare", "Schools", "Restaurants", "Universities", "Bluebikes",
]
DOMAIN_COLS = {
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
GRADE_COLS = {
    "Safety":       "SAFETY_GRADE",
    "Transit":      "TRANSIT_GRADE",
    "Housing":      "HOUSING_GRADE",
    "Grocery":      "GROCERY_GRADE",
    "Healthcare":   "HEALTHCARE_GRADE",
    "Schools":      "SCHOOL_GRADE",
    "Restaurants":  "RESTAURANT_GRADE",
    "Universities": "EDUCATION_GRADE",
    "Bluebikes":    "BIKESHARE_GRADE",
}


def grade_to_color(grade: str) -> str:
    g = str(grade).upper().strip()
    if g in ("EXCELLENT", "WELL_STOCKED", "AFFORDABLE"):
        return C_GREEN
    elif g in ("GOOD", "ADEQUATE"):
        return C_TEAL
    elif g in ("MODERATE", "AVERAGE", "PREMIUM"):
        return C_AMBER
    elif g in ("LIMITED", "HIGH CONCERN", "BELOW_AVERAGE", "FOOD_DESERT"):
        return C_RED
    return C_SUBTEXT


def grade_to_bg(grade: str) -> str:
    g = str(grade).upper().strip()
    if g in ("EXCELLENT", "WELL_STOCKED", "AFFORDABLE"):
        return C_GREEN_LIGHT
    elif g in ("GOOD", "ADEQUATE"):
        return "#CCFBF1"
    elif g in ("MODERATE", "AVERAGE", "PREMIUM"):
        return C_AMBER_LIGHT
    elif g in ("LIMITED", "HIGH CONCERN", "BELOW_AVERAGE", "FOOD_DESERT"):
        return C_RED_LIGHT
    return C_GRID


# ── Text helpers ───────────────────────────────────────────────────────────────
def trim_to_last_sentence(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if re.search(r'[.!?]\s*$', text):
        return text
    last = max(text.rfind('. '), text.rfind('.\n'), text.rfind('.'),
               text.rfind('! '), text.rfind('!\n'), text.rfind('!'),
               text.rfind('? '), text.rfind('?\n'), text.rfind('?'))
    if last > len(text) * 0.5:
        return text[:last + 1].strip()
    return text


def format_numbered_text(text: str) -> list:
    if not text:
        return [text]
    cleaned = re.sub(r'Paragraph\s*\d+\s*:\s*', '', text, flags=re.IGNORECASE)
    pattern = r'(?:^|\n)\s*(?:\d+[.)]\s*|\(\d+\)\s*)'
    parts   = re.split(pattern, cleaned.strip())
    parts   = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in cleaned.strip().split("\n\n") if p.strip()]
    if len(parts) <= 1:
        return [cleaned.strip()]
    return parts


# ── Snowflake helpers ──────────────────────────────────────────────────────────
def get_conn():
    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT, user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD, warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE, role=SNOWFLAKE_ROLE, schema="MARTS",
    )


def run_query(sql: str, conn, params: Optional[tuple] = None) -> pd.DataFrame:
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, params) if params else cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        if cur is not None:
            cur.close()


def cortex_complete(prompt: str, conn) -> str:
    cur = None
    try:
        safe = prompt.replace("'", "\\'")
        cur  = conn.cursor()
        cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{safe}')")
        row = cur.fetchone()
        return row[0].strip() if row else ""
    except Exception as e:
        print(f"[Report] Cortex failed: {e}")
        return ""
    finally:
        if cur is not None:
            cur.close()


# ── Data fetchers (used in standalone mode + router fallback) ──────────────────
def fetch_domain_data(neighborhood: str, conn) -> dict:
    name = neighborhood.strip().upper()
    sql  = """
        SELECT
            ml.NEIGHBORHOOD_NAME, ml.CITY, ml.SQMILES,
            nms.MASTER_SCORE, nms.MASTER_GRADE,
            nms.TOP_STRENGTH, nms.TOP_WEAKNESS,
            ns.SAFETY_SCORE,  ns.SAFETY_GRADE,  ns.SAFETY_DESCRIPTION,
            nm.TRANSIT_SCORE, nm.TRANSIT_GRADE, nm.TRANSIT_DESCRIPTION,
            nho.HOUSING_SCORE, nho.HOUSING_GRADE,
            ng.GROCERY_SCORE,  ng.GROCERY_GRADE,
            nh.HEALTHCARE_SCORE, nh.HEALTHCARE_GRADE,
            nsc.SCHOOL_SCORE, nsc.SCHOOL_GRADE,
            nr.RESTAURANT_SCORE, nr.RESTAURANT_GRADE, nr.RESTAURANT_DESCRIPTION,
            nu.EDUCATION_SCORE, nu.EDUCATION_GRADE,  nu.EDUCATION_DESCRIPTION,
            nb.BIKESHARE_SCORE, nb.BIKESHARE_GRADE
        FROM MARTS.MASTER_LOCATION ml
        LEFT JOIN NEIGHBOURWISE_DOMAINS.ANALYTICS.NEIGHBORHOOD_MASTER_SCORE nms
            ON ml.LOCATION_ID = nms.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_SAFETY ns
            ON ml.LOCATION_ID = ns.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_MBTA nm
            ON ml.LOCATION_ID = nm.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_HOUSING nho
            ON ml.LOCATION_ID = nho.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_GROCERY_STORES ng
            ON ml.LOCATION_ID = ng.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_HEALTHCARE nh
            ON ml.LOCATION_ID = nh.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_SCHOOLS nsc
            ON ml.LOCATION_ID = nsc.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_RESTAURANTS nr
            ON ml.LOCATION_ID = nr.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_UNIVERSITIES nu
            ON ml.LOCATION_ID = nu.LOCATION_ID
        LEFT JOIN MARTS.MRT_NEIGHBORHOOD_BLUEBIKES nb
            ON ml.LOCATION_ID = nb.LOCATION_ID
        WHERE UPPER(ml.NEIGHBORHOOD_NAME) = %s
        LIMIT 1
    """
    df = run_query(sql, conn, (name,))
    return df.iloc[0].to_dict() if not df.empty else {}


def fetch_neighboring_neighborhoods(
    neighborhood: str, city: str, conn
) -> pd.DataFrame:
    name       = neighborhood.strip().upper()
    city_upper = city.strip().upper()
    group_filter = (
        "ml.IS_BOSTON = TRUE"      if city_upper == "BOSTON"    else
        "ml.IS_CAMBRIDGE = TRUE"   if city_upper == "CAMBRIDGE" else
        "ml.IS_GREATER_BOSTON = TRUE"
    )
    score_df = run_query(
        """SELECT MASTER_SCORE
           FROM NEIGHBOURWISE_DOMAINS.ANALYTICS.NEIGHBORHOOD_MASTER_SCORE
           WHERE UPPER(NEIGHBORHOOD_NAME) = %s LIMIT 1""",
        conn, (name,)
    )
    if score_df.empty or score_df.iloc[0]["MASTER_SCORE"] is None:
        return pd.DataFrame()
    target = float(score_df.iloc[0]["MASTER_SCORE"])
    return run_query(
        f"""SELECT nms.NEIGHBORHOOD_NAME, nms.MASTER_SCORE, nms.MASTER_GRADE
            FROM NEIGHBOURWISE_DOMAINS.ANALYTICS.NEIGHBORHOOD_MASTER_SCORE nms
            JOIN MARTS.MASTER_LOCATION ml ON nms.LOCATION_ID = ml.LOCATION_ID
            WHERE {group_filter}
              AND UPPER(nms.NEIGHBORHOOD_NAME) != %s
              AND nms.MASTER_SCORE IS NOT NULL
            ORDER BY ABS(nms.MASTER_SCORE - %s) ASC
            LIMIT 5""",
        conn, (name, target)
    )


def fetch_crime_trend(neighborhood: str, conn) -> pd.DataFrame:
    return run_query(
        """SELECT YEAR_MONTH, COUNT(*) AS CRIME_COUNT
           FROM MARTS.MRT_BOSTON_CRIME
           WHERE UPPER(NEIGHBORHOOD_NAME) = %s
             AND YEAR_MONTH <= '2026-02'
             AND YEAR_MONTH >= '2025-03'
           GROUP BY YEAR_MONTH ORDER BY YEAR_MONTH ASC""",
        conn, (neighborhood.strip().upper(),)
    )


def fetch_sarimax_forecast(neighborhood: str, conn) -> pd.DataFrame:
    try:
        return run_query(
            """SELECT FORECAST_MONTH, FORECASTED_COUNT, LOWER_CI, UPPER_CI,
                      ARIMA_ORDER, TRAIN_MAPE
               FROM NEIGHBOURWISE_DOMAINS.CRIME_ANALYSIS.CA_CRIME_FORECAST
               WHERE UPPER(NEIGHBORHOOD_NAME) = %s
               ORDER BY FORECAST_MONTH ASC""",
            conn, (neighborhood.strip().upper(),)
        )
    except Exception as e:
        print(f"[Report] SARIMAX fetch failed (non-critical): {e}")
        return pd.DataFrame()


def fetch_crime_narrative(neighborhood: str, conn) -> dict:
    try:
        df = run_query(
            """SELECT RECENT_TREND, RECENT_AVG_MONTHLY, FORECAST_MONTH,
                      FORECASTED_COUNT, TRAIN_MAPE, N_HOTSPOT_CLUSTERS,
                      SAFETY_NARRATIVE, RELIABILITY_FLAG
               FROM NEIGHBOURWISE_DOMAINS.CRIME_ANALYSIS.CA_CRIME_SAFETY_NARRATIVE
               WHERE UPPER(NEIGHBORHOOD_NAME) = %s LIMIT 1""",
            conn, (neighborhood.strip().upper(),)
        )
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        print(f"[Report] Crime narrative fetch failed (non-critical): {e}")
        return {}


def fetch_rag_context(neighborhood: str, conn) -> list:
    try:
        query  = f"neighborhood character lifestyle safety restaurants transit {neighborhood}"
        safe_q = query.replace("'", "\\'")
        sql    = f"""
            SELECT CHUNK_TEXT,
                   VECTOR_COSINE_SIMILARITY(
                       CHUNK_EMBEDDING,
                       SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 'query: {safe_q}')
                   ) AS SCORE
            FROM NEIGHBOURWISE_DOMAINS.RAW_UNSTRUCTURED.RAW_DOMAIN_CHUNKS
            WHERE SCORE > 0.60
            ORDER BY SCORE DESC LIMIT 5
        """
        df = run_query(sql, conn)
        return df["CHUNK_TEXT"].tolist() if not df.empty else []
    except Exception as e:
        print(f"[Report] RAG fetch failed (non-critical): {e}")
        return []


def clean_rag_chunks(chunks: list) -> list:
    cleaned = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue
        text  = re.sub(r'---\s*Page\s*\d+\s*---', '', chunk.strip())
        text  = re.sub(r'\(cont\.\)', '', text)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        text  = re.sub(r'[■□●▪▫]+', ' ', ' '.join(lines))
        text  = re.sub(r'\s{2,}', ' ', text).strip()
        if len(text) > 80:
            cleaned.append(text)
    return cleaned


def summarize_rag_with_cortex(
    chunks: list, neighborhood: str, city: str, conn
) -> str:
    if not chunks:
        return ""
    combined = "\n\n".join(chunks[:3])[:2000]
    prompt   = (
        f"You are writing the 'Lifestyle & Character' section of a neighborhood report "
        f"for {neighborhood} in {city}, Massachusetts. "
        f"Synthesize the excerpts below into 2 concise paragraphs describing the "
        f"neighborhood's character, daily life, and what makes it unique. "
        f"Write in a confident, magazine-editorial tone. "
        f"Do NOT mention page numbers, data sources, or document references. "
        f"Do NOT use bullet points. Be specific to {neighborhood}.\n\n"
        f"Source excerpts:\n{combined}"
    )
    return trim_to_last_sentence(cortex_complete(prompt, conn))


def generate_cortex_narratives(data: dict, conn) -> dict:
    name   = data.get("NEIGHBORHOOD_NAME", "this neighborhood")
    city   = data.get("CITY", "Boston")
    scores = ", ".join(
        f"{d}: {data.get(c, 'N/A')}" for d, c in DOMAIN_COLS.items()
    )

    exec_prompt = (
        f"Write a 3-paragraph executive summary for {name} in {city}, Massachusetts "
        f"as a neighborhood recommendation report. "
        f"Domain scores (0-100): {scores}. "
        f"Master score: {data.get('MASTER_SCORE','N/A')}/100 ({data.get('MASTER_GRADE','')}). "
        f"Top strength: {data.get('TOP_STRENGTH','N/A')}. "
        f"Top weakness: {data.get('TOP_WEAKNESS','N/A')}. "
        f"Cover overall character, who it suits best, and one key trade-off. "
        f"Magazine-editorial voice. Each paragraph ends with a complete sentence. "
        f"IMPORTANT: {name} is in {city}, Massachusetts."
    )
    rec_prompt = (
        f"Based on these scores for {name} in {city}, Massachusetts: {scores}. "
        f"Write exactly 2 points:\n"
        f"1. Which persona (families, students, young professionals, retirees) "
        f"this neighborhood suits best and why (2-3 sentences).\n"
        f"2. Which persona should look elsewhere and why (2-3 sentences).\n"
        f"Format as:\n1. [first point]\n2. [second point]\n"
        f"Do NOT write 'Paragraph 1'. Just use 1. and 2. Be direct and specific."
    )

    # ── Per-domain narrative prompts ──────────────────────────────────────────
    domain_prompts = {
        "Housing": (
            f"Write 2 sentences about housing in {name}, {city}, Massachusetts. "
            f"Housing score: {data.get('HOUSING_SCORE','N/A')}/100, "
            f"grade: {data.get('HOUSING_GRADE','N/A')}. "
            f"Cover affordability, property types, and what renters or buyers can expect. "
            f"Be specific and factual. End with a complete sentence."
        ),
        "Grocery": (
            f"Write 2 sentences about grocery and food access in {name}, {city}, Massachusetts. "
            f"Grocery score: {data.get('GROCERY_SCORE','N/A')}/100, "
            f"grade: {data.get('GROCERY_GRADE','N/A')}. "
            f"Cover store availability, food desert risk, and daily convenience. "
            f"Be specific and factual. End with a complete sentence."
        ),
        "Healthcare": (
            f"Write 2 sentences about healthcare access in {name}, {city}, Massachusetts. "
            f"Healthcare score: {data.get('HEALTHCARE_SCORE','N/A')}/100, "
            f"grade: {data.get('HEALTHCARE_GRADE','N/A')}. "
            f"Cover facility availability, hospitals, clinics, and access quality. "
            f"Be specific and factual. End with a complete sentence."
        ),
        "Schools": (
            f"Write 2 sentences about schools in {name}, {city}, Massachusetts. "
            f"School score: {data.get('SCHOOL_SCORE','N/A')}/100, "
            f"grade: {data.get('SCHOOL_GRADE','N/A')}. "
            f"Cover public vs private options, school levels, and education quality. "
            f"Be specific and factual. End with a complete sentence."
        ),
        "Bluebikes": (
            f"Write 2 sentences about Bluebikes bikeshare in {name}, {city}, Massachusetts. "
            f"Bluebikes score: {data.get('BIKESHARE_SCORE','N/A')}/100, "
            f"grade: {data.get('BIKESHARE_GRADE','N/A')}. "
            f"Cover station availability, dock capacity, and cycling convenience. "
            f"Be specific and factual. End with a complete sentence."
        ),
    }

    narratives = {
        "executive_summary": trim_to_last_sentence(cortex_complete(exec_prompt, conn)),
        "recommendation":    trim_to_last_sentence(cortex_complete(rec_prompt, conn)),
    }

    # Generate missing domain narratives
    for domain, prompt in domain_prompts.items():
        print(f"[Report] Generating {domain} narrative...")
        narratives[domain] = trim_to_last_sentence(cortex_complete(prompt, conn))

    return narratives

# ── Chart generation (STANDALONE MODE ONLY) ────────────────────────────────────
# Only called when report_agent is run directly without precomputed charts.
# When called via router, graphic_agent.py generates these instead.

def _generate_charts_standalone(
    data: dict, neighbor_df: pd.DataFrame,
    crime_df: pd.DataFrame, forecast_df: pd.DataFrame,
    neighborhood: str
) -> dict:
    """
    Fallback chart generation for standalone mode.
    Imports from graphic_agent so chart style is always consistent —
    report_agent no longer has its own chart generation logic.
    """
    print("[Report] Standalone mode — generating charts via GraphicAgent...")
    try:
        from graphic_agent import (
            generate_radar_chart,
            generate_bar_neighbors,
            generate_grouped_bar,
            generate_crime_trend,
        )
        return {
            "chart_radar":
                generate_radar_chart(data, neighborhood),
            "chart_bar_neighbors":
                generate_bar_neighbors(data, neighbor_df, neighborhood),
            "chart_grouped_bar":
                generate_grouped_bar(data, neighborhood),
            "chart_crime_trend":
                generate_crime_trend(crime_df, neighborhood, forecast_df=forecast_df),
        }
    except ImportError as e:
        print(f"[Report] GraphicAgent import failed: {e}")
        return {}


# ── Custom ReportLab Flowable ──────────────────────────────────────────────────
class ColorBand(Flowable):
    def __init__(self, text, width, height=28, bg_color=C_ACCENT,
                 text_color="#FFFFFF", font_name="Helvetica-Bold", font_size=14):
        super().__init__()
        self.text = text; self.w = width; self.h = height
        self.bg = bg_color; self.tc = text_color
        self.fn = font_name; self.fs = font_size

    def wrap(self, aW, aH):
        return self.w, self.h

    def draw(self):
        self.canv.setFillColor(colors.HexColor(self.bg))
        self.canv.roundRect(0, 0, self.w, self.h, 4, fill=1, stroke=0)
        self.canv.setFillColor(colors.HexColor(self.tc))
        self.canv.setFont(self.fn, self.fs)
        self.canv.drawString(12, (self.h - self.fs) / 2 + 1, self.text)


# ── PDF builder ────────────────────────────────────────────────────────────────
def build_pdf(
    neighborhood: str,
    data: dict,
    neighbor_df: pd.DataFrame,
    narratives: dict,
    rag_narrative: str,
    chart_paths: dict,
    pdf_path: str,
    crime_narrative: dict = None,
    image_paths: list = None,
) -> dict:
    """
    Assembles the PDF from pre-generated charts and fetched data.
    chart_paths must contain file paths to already-rendered PNG files.
    This function does NO chart generation — it only reads chart files.
    """
    page_w, page_h = letter
    margin    = 0.75 * inch
    content_w = page_w - 2 * margin

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=0.6 * inch,
        title=f"NeighbourWise AI — {neighborhood}",
        author="NeighbourWise AI",
    )

    # ── Styles ─────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    H2      = ParagraphStyle("H2", parent=base["Heading2"],
                              fontSize=15, textColor=colors.HexColor(C_ACCENT),
                              spaceBefore=14, spaceAfter=6,
                              fontName="Helvetica-Bold", leading=18)
    BODY    = ParagraphStyle("BODY", parent=base["Normal"],
                              fontSize=10.5, leading=16,
                              textColor=colors.HexColor(C_PRIMARY),
                              fontName="Helvetica", spaceAfter=8,
                              alignment=TA_JUSTIFY)
    SMALL   = ParagraphStyle("SMALL", parent=base["Normal"],
                              fontSize=8.5, leading=12,
                              textColor=colors.HexColor(C_SUBTEXT),
                              fontName="Helvetica", spaceAfter=4)
    CAPTION = ParagraphStyle("CAPTION", parent=SMALL,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
    QUOTE   = ParagraphStyle("QUOTE", parent=base["Normal"],
                              fontSize=11, leading=17,
                              textColor=colors.HexColor(C_ACCENT_DARK),
                              fontName="Helvetica", spaceAfter=10,
                              leftIndent=12, rightIndent=12)
    FOOTER  = ParagraphStyle("FOOTER", parent=SMALL, alignment=TA_CENTER,
                              textColor=colors.HexColor(C_SUBTEXT))

    COVER_TITLE = ParagraphStyle("COVER_TITLE", parent=base["Title"],
                                  fontSize=36, textColor=colors.HexColor(C_PRIMARY),
                                  alignment=TA_LEFT, fontName="Helvetica-Bold",
                                  spaceAfter=4, leading=40)
    COVER_SUB   = ParagraphStyle("COVER_SUB", parent=base["Normal"],
                                  fontSize=14, textColor=colors.HexColor(C_SUBTEXT),
                                  alignment=TA_LEFT, fontName="Helvetica",
                                  spaceAfter=2, leading=18)
    COVER_SCORE = ParagraphStyle("COVER_SCORE", parent=base["Normal"],
                                  fontSize=64, textColor=colors.HexColor(C_ACCENT),
                                  alignment=TA_LEFT, fontName="Helvetica-Bold",
                                  spaceAfter=0, leading=68)

    story          = []
    sections_built = {}

    def section_header(title: str, color: str = C_ACCENT):
        story.append(Spacer(1, 0.12 * inch))
        story.append(ColorBand(
            title.upper(), content_w, height=30,
            bg_color=color, text_color="#FFFFFF",
            font_name="Helvetica-Bold", font_size=13,
        ))
        story.append(Spacer(1, 0.12 * inch))

    def smart_break(min_inches: float = 3.0):
        story.append(CondPageBreak(min_inches * inch))

    def add_chart(key: str, label: str, caption: str = "",
                  width_in: float = 6.0, aspect: float = 0.55):
        """
        Add a pre-generated chart PNG to the PDF.
        Only reads from chart_paths — never generates.
        """
        path = chart_paths.get(key)
        if path and Path(path).exists():
            story.append(Paragraph(f"<b>{label}</b>", H2))
            story.append(Image(path, width=width_in * inch,
                               height=width_in * inch * aspect))
            if caption:
                story.append(Paragraph(caption, CAPTION))
            sections_built[key] = True
        else:
            print(f"[Report] Chart '{key}' not found at path: {path} — skipping")
            sections_built[key] = False

    # ══ COVER ════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.6 * inch))
    story.append(HRFlowable(width="35%", thickness=4,
                             color=colors.HexColor(C_ACCENT), hAlign="LEFT"))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("NEIGHBOURWISE AI", ParagraphStyle(
        "BRAND", parent=SMALL, fontSize=11,
        textColor=colors.HexColor(C_ACCENT), fontName="Helvetica-Bold"
    )))
    story.append(Paragraph("Neighborhood Intelligence Report", COVER_SUB))
    story.append(Spacer(1, 0.5 * inch))

    display_name = str(data.get("NEIGHBORHOOD_NAME", neighborhood)).title()
    city_name    = str(data.get("CITY", "Boston")).title()
    story.append(Paragraph(display_name, COVER_TITLE))
    story.append(Paragraph(f"{city_name}, Massachusetts", COVER_SUB))
    story.append(Spacer(1, 0.4 * inch))

    master_score = data.get("MASTER_SCORE")
    if master_score is not None:
        story.append(Paragraph(f"{float(master_score):.1f}", COVER_SCORE))
        story.append(Paragraph(
            f'<font color="{C_SUBTEXT}">out of 100</font>',
            ParagraphStyle("GL", parent=COVER_SUB, fontSize=16)
        ))

    story.append(Spacer(1, 0.3 * inch))

    # Strength / Weakness — clean text, no pills
    top_s = str(data.get("TOP_STRENGTH", "") or "").title()
    top_w = str(data.get("TOP_WEAKNESS",  "") or "").title()
    if top_s:
        story.append(Paragraph(
            f'<font color="{C_GREEN}">▲ Strength: </font><b>{top_s}</b>',
            ParagraphStyle("SW", parent=BODY, fontSize=12, spaceAfter=4)
        ))
    if top_w:
        story.append(Paragraph(
            f'<font color="{C_RED}">▼ Weakness: </font><b>{top_w}</b>',
            ParagraphStyle("SW2", parent=BODY, fontSize=12, spaceAfter=4)
        ))

    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  ·  "
        f"DAMG7374  ·  NeighbourWise AI", FOOTER
    ))
    story.append(PageBreak())
    sections_built["cover"] = True

    # ══ EXECUTIVE SUMMARY ════════════════════════════════════════════════════
    section_header("Executive Summary")
    exec_text = narratives.get("executive_summary", "")
    for para in exec_text.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, BODY))
    sections_built["executive_summary"] = bool(exec_text)

    # ══ DOMAIN SCORECARD ═════════════════════════════════════════════════════
    smart_break(4.0)
    section_header("Domain Scorecard", C_PRIMARY)

    tbl_data = [[
        Paragraph("<b>DOMAIN</b>", ParagraphStyle(
            "TH", parent=SMALL, textColor=colors.HexColor("#FFFFFF"))),
        Paragraph("<b>SCORE</b>", ParagraphStyle(
            "TH2", parent=SMALL, textColor=colors.HexColor("#FFFFFF"),
            alignment=TA_CENTER)),
        Paragraph("<b>GRADE</b>", ParagraphStyle(
            "TH3", parent=SMALL, textColor=colors.HexColor("#FFFFFF"),
            alignment=TA_CENTER)),
    ]]
    for domain in DOMAIN_ORDER:
        score  = data.get(DOMAIN_COLS[domain])
        grade  = str(data.get(GRADE_COLS[domain]) or "—")
        sc_str = f"{float(score):.1f}" if score is not None else "—"
        gc     = grade_to_color(grade)
        tbl_data.append([
            Paragraph(
                f'<font color="{DOMAIN_COLORS.get(domain, C_PRIMARY)}">●</font>'
                f'  <b>{domain}</b>', BODY
            ),
            Paragraph(f'<font color="{gc}"><b>{sc_str}</b></font>',
                      ParagraphStyle("SC", parent=BODY, alignment=TA_CENTER)),
            Paragraph(f'<font color="{gc}"><b>{grade.replace("_"," ").title()}</b></font>',
                      ParagraphStyle("GR", parent=BODY, alignment=TA_CENTER)),
        ])

    scorecard = Table(tbl_data, colWidths=[2.8*inch, 1.8*inch, 2.4*inch])
    scorecard.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(scorecard)
    sections_built["domain_scorecard"] = True

    # ══ VISUAL ANALYSIS — uses pre-generated chart files ═════════════════════
    smart_break(5.0)
    section_header("Visual Analysis")
    add_chart("chart_radar", "Domain Radar",
              "Each spoke represents one of the 9 scored domains (0–100).",
              width_in=4.5, aspect=0.85)

    smart_break(4.5)
    add_chart("chart_grouped_bar", "All Domain Scores",
              "Color-coded by domain  ·  Higher = stronger performance.",
              width_in=6.2)

    # ══ COMPARISON ═══════════════════════════════════════════════════════════
    smart_break(4.0)
    section_header("Comparison & Trends")
    add_chart("chart_bar_neighbors", "Neighborhood Comparison",
              "Blue bar = this neighborhood  ·  vs 5 nearest by master score.",
              width_in=6.2)

    # ══ CRIME TREND ══════════════════════════════════════════════════════════
    smart_break(4.0)
    add_chart("chart_crime_trend", "Crime Trend & Forecast",
              "Blue = actual  ·  Orange dashed = SARIMAX forecast + CI band.",
              width_in=6.2)

    if crime_narrative and crime_narrative.get("SAFETY_NARRATIVE"):
        narr_text = str(crime_narrative["SAFETY_NARRATIVE"]).strip()
        trend     = str(crime_narrative.get("RECENT_TREND", "")).title()
        mape      = crime_narrative.get("TRAIN_MAPE")
        rel       = str(crime_narrative.get("RELIABILITY_FLAG", ""))
        meta      = "  ·  ".join(filter(None, [
            f"Trend: {trend}" if trend else "",
            f"Model MAPE: {mape:.1f}%" if mape is not None else "",
            f"Reliability: {rel}" if rel else "",
        ]))
        story.append(Paragraph(
            f'<font color="{C_SUBTEXT}"><i>{meta}</i></font>', SMALL
        ))
        story.append(Paragraph(narr_text, BODY))

# ══ DOMAIN NARRATIVES ════════════════════════════════════════════════════
    smart_break(3.5)
    section_header("Domain Analysis", C_SECONDARY)

    # Domains with pre-existing DB descriptions
    db_narrative_map = {
        "Safety":       "SAFETY_DESCRIPTION",
        "Transit":      "TRANSIT_DESCRIPTION",
        "Restaurants":  "RESTAURANT_DESCRIPTION",
        "Universities": "EDUCATION_DESCRIPTION",
    }
    # Domains with Cortex-generated narratives
    cortex_narrative_domains = [
        "Housing", "Grocery", "Healthcare", "Schools", "Bluebikes"
    ]

    any_narrative = False
    for domain in DOMAIN_ORDER:
        dc   = DOMAIN_COLORS.get(domain, C_ACCENT)

        # Get text — prefer DB description, fall back to Cortex-generated
        if domain in db_narrative_map:
            text = str(data.get(db_narrative_map[domain]) or "").strip()
        else:
            text = str(narratives.get(domain) or "").strip()

        if not text:
            continue

        block = [
            Paragraph(
                f'<font color="{dc}">●</font>  <b>{domain}</b>', H2
            ),
            Paragraph(text, BODY),
        ]
        story.append(KeepTogether(block))
        any_narrative = True

    sections_built["domain_narratives"] = any_narrative

    # ══ LIFESTYLE CONTEXT ════════════════════════════════════════════════════
    smart_break(3.0)
    section_header("Lifestyle & Character", C_TEAL)
    if rag_narrative:
        for para in rag_narrative.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, BODY))
        sections_built["lifestyle_context"] = True
    else:
        story.append(Paragraph(
            "Lifestyle context not available — run neighbourwise_rag.py "
            "to populate this section.", SMALL
        ))
        sections_built["lifestyle_context"] = False

    # ══ NEIGHBORHOOD IMAGES ══════════════════════════════════════════════════
    valid_images = [p for p in (image_paths or []) if p and Path(p).exists()]
    if valid_images:
        smart_break(3.0)
        section_header("Neighborhood Visuals", C_INDIGO)

        # Display 2 images per row
        img_pairs = [valid_images[i:i+2] for i in range(0, len(valid_images), 2)]
        for pair in img_pairs:
            row_images = []
            for img_path in pair:
                cell = [Image(img_path, width=3.4 * inch, height=2.0 * inch)]
                row_images.append(cell)

            # Pad to 2 columns if odd number
            if len(row_images) == 1:
                row_images.append([""])

            img_tbl = Table(
                [row_images],
                colWidths=[3.6 * inch, 3.6 * inch],
            )
            img_tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(img_tbl)

        sections_built["neighborhood_images"] = True
    else:
        sections_built["neighborhood_images"] = False

    # ══ RECOMMENDATION ═══════════════════════════════════════════════════════
    smart_break(3.0)
    section_header("Who Should Live Here?", C_ACCENT_DARK)
    rec_text = narratives.get("recommendation", "").strip()
    if rec_text:
        items      = format_numbered_text(rec_text)
        quote_cell = []
        for i, item in enumerate(items):
            item = item.strip()
            if not item:
                continue
            if len(items) > 1:
                prefix = f'<font color="{C_ACCENT}"><b>{i+1}.</b></font>  '
                quote_cell.append(Paragraph(prefix + item, QUOTE))
                quote_cell.append(Spacer(1, 0.06 * inch))
            else:
                quote_cell.append(Paragraph(item, QUOTE))

        rec_tbl = Table(
            [[quote_cell]], colWidths=[content_w - 0.2 * inch]
        )
        rec_tbl.setStyle(TableStyle([
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
            ("TOPPADDING",    (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("BACKGROUND",    (0, 0), (-1, -1),
             colors.HexColor(C_ACCENT_LIGHT)),
        ]))
        story.append(rec_tbl)
    else:
        story.append(Paragraph("Recommendation not available.", SMALL))
    sections_built["recommendation"] = bool(rec_text)

    # ══ FOOTER ═══════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=0.4,
                             color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "Data: Analyze Boston API · Cambridge Open Data · FBI Crime Data "
        "Explorer · MBTA GTFS · Snowflake Cortex · NeighbourWise RAG  ·  "
        "Scores normalized 0–100 across 51 neighborhoods  ·  "
        f"Generated {datetime.now().strftime('%B %d, %Y')}  ·  DAMG7374",
        FOOTER
    ))

    doc.build(story)
    print(f"[Report] PDF built: {pdf_path}")
    return sections_built


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    neighborhood: str,
    precomputed: dict = None,
) -> str:
    """
    Generate a full PDF report for a neighborhood.

    precomputed (optional, from router):
    {
        "data":            dict,
        "neighbor_df":     DataFrame,
        "crime_df":        DataFrame,
        "forecast_df":     DataFrame,
        "crime_narrative": dict,
        "rag_narrative":   str,
        "narratives":      dict,
        "chart_paths":     dict,   ← generated by GraphicAgent
        "image_paths":     list,   ← generated by GraphicAgent (DALL-E)
    }

    If precomputed is None or a key is missing → fetches/generates internally.
    """
    print(f"\n{'='*60}")
    print(f"NeighbourWise Report Agent (v3 — Assembly Edition)")
    print(f"Neighborhood : {neighborhood}")
    print(f"Mode         : {'Router (precomputed inputs)' if precomputed else 'Standalone'}")
    print(f"{'='*60}\n")

    conn      = get_conn()
    validator = UniversalValidator(conn)

    try:
        # ── Checkpoint 1: neighborhood valid ─────────────────────────────────
        print("[Report] Checkpoint 1 — neighborhood validation...")
        pre = validator.validate(
            AgentType.REPORT,
            {
                "checkpoint":   "pre_fetch",
                "neighborhood": neighborhood,
            }
        )
        pre.print_summary()
        if not pre.passed:
            print("❌ Report aborted — neighborhood validation failed.")
            return ""

        pc = precomputed or {}

        # ── Fetch domain data ─────────────────────────────────────────────────
        data = pc.get("data") or fetch_domain_data(neighborhood, conn)
        if not data:
            print(f"❌ No data found for '{neighborhood}'")
            return ""

        city = str(data.get("CITY", "Boston")).title()

        # ── Fetch supporting data (use precomputed if available) ──────────────
        neighbor_df = (
            pc.get("neighbor_df")
            if pc.get("neighbor_df") is not None
            else fetch_neighboring_neighborhoods(neighborhood, city, conn)
        )
        crime_df = (
            pc.get("crime_df")
            if pc.get("crime_df") is not None
            else fetch_crime_trend(neighborhood, conn)
        )
        forecast_df = (
            pc.get("forecast_df")
            if pc.get("forecast_df") is not None
            else fetch_sarimax_forecast(neighborhood, conn)
        )
        crime_narrative = (
            pc.get("crime_narrative")
            if pc.get("crime_narrative") is not None
            else fetch_crime_narrative(neighborhood, conn)
        )

        # ── RAG narrative ─────────────────────────────────────────────────────
        rag_narrative = pc.get("rag_narrative")
        if rag_narrative is None:
            print("[Report] Fetching RAG context...")
            raw_rag       = fetch_rag_context(neighborhood, conn)
            cleaned_rag   = clean_rag_chunks(raw_rag)
            rag_narrative = summarize_rag_with_cortex(
                cleaned_rag, neighborhood, city, conn
            )

        # ── Cortex narratives ─────────────────────────────────────────────────
        narratives = pc.get("narratives")
        if not narratives:
            print("[Report] Generating Cortex narratives...")
            narratives = generate_cortex_narratives(data, conn)

        # Inject NARRATIVES key for validator
        data["NARRATIVES"] = {
            k: data.get(v, "")
            for k, v in {
                "Safety":       "SAFETY_DESCRIPTION",
                "Transit":      "TRANSIT_DESCRIPTION",
                "Restaurants":  "RESTAURANT_DESCRIPTION",
                "Universities": "EDUCATION_DESCRIPTION",
            }.items()
        }

        # ── Checkpoint 2: data completeness ───────────────────────────────────
        raw_rag_for_validator = (
            fetch_rag_context(neighborhood, conn)
            if not pc.get("rag_narrative")
            else ["(precomputed)"]
        )
        print("\n[Report] Checkpoint 2 — data completeness validation...")
        data_result = validator.validate(
            AgentType.REPORT,
            {
                "checkpoint":   "post_fetch",
                "neighborhood": neighborhood,
                "data":         data,
                "neighbor_df":  neighbor_df,
                "crime_df":     crime_df,
                "rag_results":  raw_rag_for_validator,
            }
        )
        data_result.print_summary()

        # ── Chart paths ───────────────────────────────────────────────────────
        chart_paths = pc.get("chart_paths")
        if not chart_paths:
            chart_paths = _generate_charts_standalone(
                data, neighbor_df, crime_df, forecast_df, neighborhood
            )

        # ── DALL-E image paths ────────────────────────────────────────────────
        # Use precomputed if provided by router, otherwise generate in standalone
        image_paths = pc.get("image_paths")
        if not image_paths:
            print(f"\n[Report] Generating DALL-E neighborhood images...")
            try:
                from graphic_agent import generate_neighborhood_images
                image_paths = generate_neighborhood_images(neighborhood)
                # Filter to only paths that actually exist on disk
                image_paths = [p for p in (image_paths or []) if p and Path(p).exists()]
                print(f"[Report] DALL-E: {len(image_paths)}/4 images generated")
            except Exception as e:
                print(f"[Report] ⚠️  DALL-E image generation failed: {e} — continuing without images")
                image_paths = []

        # Validate all chart paths exist
        for key, path in chart_paths.items():
            if path and not Path(path).exists():
                print(f"[Report] ⚠️  Chart file missing: {key} → {path}")

        # ── Build PDF ─────────────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", neighborhood.lower())
        pdf_path  = str(OUTPUT_BASE / f"{safe_name}_report_{timestamp}.pdf")

        print(f"\n[Report] Assembling PDF...")
        sections_built = build_pdf(
            neighborhood=neighborhood,
            data=data,
            neighbor_df=neighbor_df,
            narratives=narratives,
            rag_narrative=rag_narrative,
            chart_paths=chart_paths,
            pdf_path=pdf_path,
            crime_narrative=crime_narrative,
            image_paths=image_paths,
        )

        # ── Checkpoint 3: report completeness ─────────────────────────────────
        print("\n[Report] Checkpoint 3 — report completeness validation...")
        report_result = validator.validate(
            AgentType.REPORT,
            {
                "checkpoint":        "post_build",
                "neighborhood":      neighborhood,
                "report_sections":   sections_built,
                "chart_paths":       chart_paths,
                "pdf_path":          pdf_path,
                "executive_summary": narratives.get("executive_summary", ""),
            }
        )
        report_result.print_summary()

        print(f"\n{'='*60}")
        print(f"✅ Report saved: {pdf_path}")
        print(f"{'='*60}")
        return pdf_path

    except Exception as e:
        print(f"\n❌ Report generation failed:")
        traceback.print_exc()
        return ""
    finally:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 report_agent.py \"<neighborhood>\"")
        print("\nExamples:")
        print("  python3 report_agent.py \"Fenway\"")
        print("  python3 report_agent.py \"Jamaica Plain\"")
        print("  python3 report_agent.py \"Newton\"")
        sys.exit(1)

    neighborhood = " ".join(sys.argv[1:]).strip()
    result = generate_report(neighborhood)
    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()