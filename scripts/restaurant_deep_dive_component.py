"""
restaurant_deep_dive_component.py — NeighbourWise AI
══════════════════════════════════════════════════════════════════════════════
Drop-in replacement for the Restaurants domain section in neighbourwise_app.py.

Usage in neighbourwise_app.py (replace the elif domain_filter == "Restaurants" block):

    elif domain_filter == "Restaurants":
        from restaurant_deep_dive_component import render_restaurant_deep_dive
        domain_data_restaurants = load_domain("restaurants", hood_filter)
        render_restaurant_deep_dive(
            domain_data = domain_data_restaurants,
            hood_filter = hood_filter,
        )

The component handles two modes automatically:
  • ALL neighborhoods  → citywide overview: KPI ribbon, score + price charts,
                         cuisine heatmap, quality scatter, full ranked table
  • Single neighborhood → individual deep dive: KPI cards, AI narrative,
                          rating + price donuts, cuisine bar, quality radar,
                          individual restaurant cards (via /domain/restaurants
                          individual endpoint — gracefully absent if not wired)
"""

from __future__ import annotations

import logging
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / GRADE PALETTES
# ─────────────────────────────────────────────────────────────────────────────

GRADE_COLORS = {
    "EXCELLENT": "#22c55e",
    "GOOD":      "#3b82f6",
    "MODERATE":  "#f59e0b",
    "LIMITED":   "#ef4444",
}

GRADE_BG = {
    "EXCELLENT": ("rgba(34,197,94,0.15)",  "#22c55e"),
    "GOOD":      ("rgba(59,130,246,0.15)", "#3b82f6"),
    "MODERATE":  ("rgba(245,158,11,0.15)", "#f59e0b"),
    "LIMITED":   ("rgba(239,68,68,0.15)",  "#ef4444"),
}

PRICE_COLORS   = ["#14b8a6", "#fb923c", "#ec4899"]   # budget, mid-range, upscale
RATING_COLORS  = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444"]  # exc, good, avg, poor
CUISINE_COLORS = [
    "#fb923c",  # Ethnic
    "#14b8a6",  # Pizza
    "#ec4899",  # Café/Bakery
    "#22c55e",  # Fast Food
    "#3b82f6",  # Breakfast
    "#f59e0b",  # Sandwiches
    "#a78bfa",  # Healthy
    "#34d399",  # American
    "#67e8f9",  # Bar
    "#64748b",  # Other (intentionally muted)
]


# ─────────────────────────────────────────────────────────────────────────────
# INLINE CSS (injected once per render)
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Metric tiles ─────────────────────────────────────────────────────── */
.rdd-kpi {
    background: rgba(255,255,255,0.04);
    border: 1.5px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.1rem 0.85rem;
    margin-bottom: 0.6rem;
}
.rdd-kpi.accent { border-top: 2.5px solid #fb923c; }
.rdd-kpi-label {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: rgba(255,255,255,0.35); margin-bottom: 5px;
}
.rdd-kpi-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.55rem; color: #e2e8f0; line-height: 1.1;
}
.rdd-kpi-sub { font-size: 11px; color: rgba(255,255,255,0.35); margin-top: 3px; }

/* ── Narrative box ────────────────────────────────────────────────────── */
.rdd-narrative {
    background: rgba(251,146,60,0.07);
    border: 1px solid rgba(251,146,60,0.22);
    border-left: 4px solid #fb923c;
    padding: 0.85rem 1.1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.87rem; line-height: 1.7; color: #e2e8f0;
}
.rdd-narrative-title {
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem; color: #e2e8f0; margin-bottom: 5px;
}

/* ── Section card ─────────────────────────────────────────────────────── */
.rdd-section {
    background: rgba(255,255,255,0.04);
    border: 1.5px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.2rem 0.9rem;
    margin-bottom: 1rem;
}
.rdd-section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.02rem; color: #e2e8f0; margin-bottom: 2px;
}
.rdd-section-sub {
    font-size: 0.74rem; color: rgba(255,255,255,0.35); margin-bottom: 10px;
}

/* ── Grade badge ──────────────────────────────────────────────────────── */
.rdd-grade {
    display: inline-block;
    padding: 3px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
}

/* ── Stat rows ────────────────────────────────────────────────────────── */
.rdd-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.rdd-stat-row:last-child { border-bottom: none; }
.rdd-stat-label { font-size: 12px; color: rgba(255,255,255,0.42); }
.rdd-stat-val   { font-size: 12px; font-weight: 500; color: #e2e8f0; }

/* ── Restaurant cards ─────────────────────────────────────────────────── */
.rdd-rest-card {
    background: rgba(255,255,255,0.04);
    border: 1.5px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 14px 10px;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.rdd-rest-card:hover { border-color: rgba(251,146,60,0.35); }
.rdd-rest-name {
    font-weight: 600; font-size: 13px; color: #e2e8f0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 2px;
}
.rdd-rest-cuisine {
    font-size: 11px; color: rgba(255,255,255,0.35);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 7px;
}
.rdd-rest-meta { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.rdd-star { color: #f59e0b; font-size: 11px; }
.rdd-pill {
    padding: 2px 7px; border-radius: 20px;
    font-size: 10px; font-weight: 600;
}
.rdd-pill-price    { background: rgba(20,184,166,0.18); color: #14b8a6; }
.rdd-pill-delivery { background: rgba(34,197,94,0.15);  color: #22c55e; }
.rdd-pill-pickup   { background: rgba(59,130,246,0.15); color: #60a5fa; }
.rdd-rest-addr {
    font-size: 11px; color: rgba(255,255,255,0.28);
    margin-top: 5px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Summary ribbon ───────────────────────────────────────────────────── */
.rdd-ribbon {
    background: rgba(251,146,60,0.08);
    border: 1px solid rgba(251,146,60,0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    font-size: 13px; color: #e2e8f0; line-height: 1.8;
}

/* ── Ranking table ────────────────────────────────────────────────────── */
.rdd-rank-table { width: 100%; border-collapse: collapse; }
.rdd-rank-table th {
    font-size: 10px; color: rgba(255,255,255,0.3);
    text-transform: uppercase; letter-spacing: 0.06em;
    padding: 0 10px 8px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.rdd-rank-table td {
    padding: 9px 10px; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 12px; color: rgba(255,255,255,0.75);
}
.rdd-rank-table tr:hover td { background: rgba(255,255,255,0.025); }
.rdd-rank-num { color: rgba(255,255,255,0.2); font-size: 10px; width: 24px; }
.rdd-score-bar-wrap { display: flex; align-items: center; gap: 6px; }
.rdd-score-bar {
    height: 4px; background: rgba(255,255,255,0.07);
    border-radius: 2px; flex: 1; overflow: hidden; min-width: 60px;
}
.rdd-score-fill { height: 100%; background: #fb923c; border-radius: 2px; }
.rdd-score-val { font-size: 11px; font-weight: 600; color: #e2e8f0; min-width: 28px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    cls = "rdd-kpi accent" if accent else "rdd-kpi"
    return (
        f'<div class="{cls}">'
        f'<div class="rdd-kpi-label">{label}</div>'
        f'<div class="rdd-kpi-val">{value}</div>'
        f'<div class="rdd-kpi-sub">{sub}</div>'
        f'</div>'
    )


def _grade_badge(grade: str) -> str:
    bg, fg = GRADE_BG.get(grade, ("rgba(100,116,139,0.2)", "#94a3b8"))
    return (
        f'<span class="rdd-grade" '
        f'style="background:{bg};color:{fg};border:1px solid {fg}44;">'
        f'{grade}</span>'
    )


def _stat_row(label: str, val: str) -> str:
    return (
        f'<div class="rdd-stat-row">'
        f'<span class="rdd-stat-label">{label}</span>'
        f'<span class="rdd-stat-val">{val}</span>'
        f'</div>'
    )


def _section_open(title: str, subtitle: str = "") -> str:
    sub_html = f'<div class="rdd-section-sub">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="rdd-section">'
        f'<div class="rdd-section-title">{title}</div>{sub_html}'
    )


def _section_close() -> str:
    return '</div>'


def _rest_card(r: dict) -> str:
    price = r.get("price_label") or ("$" * max(1, min(r.get("price_tier", 1), 4)))
    delivery_pill = '<span class="rdd-pill rdd-pill-delivery">Delivery</span>' if r.get("has_delivery") else ""
    pickup_pill   = '<span class="rdd-pill rdd-pill-pickup">Pickup</span>'    if r.get("has_pickup")   else ""
    price_pill    = f'<span class="rdd-pill rdd-pill-price">{price}</span>'   if price                 else ""
    rating        = r.get("rating", 0)
    reviews       = r.get("reviews", 0)
    addr          = r.get("address", "")
    cuisine       = r.get("categories") or r.get("cuisine") or ""
    return (
        f'<div class="rdd-rest-card">'
        f'<div class="rdd-rest-name" title="{r["name"]}">{r["name"]}</div>'
        f'<div class="rdd-rest-cuisine">{cuisine}</div>'
        f'<div class="rdd-rest-meta">'
        f'<span style="font-size:12px;font-weight:600;color:#e2e8f0;">'
        f'<span class="rdd-star">★</span> {rating:.1f}</span>'
        f'<span style="font-size:11px;color:rgba(255,255,255,0.3);">({reviews:,})</span>'
        f'{price_pill}{delivery_pill}{pickup_pill}'
        f'</div>'
        f'<div class="rdd-rest-addr">{addr}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# ALTAIR CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(
    df: pd.DataFrame,
    x_col: str, y_col: str,
    x_title: str, color: str = "#fb923c",
    x_max: float = 100, height: int = 480,
    tooltip_extras: list = None,
):
    tips = [
        alt.Tooltip(f"{y_col}:N"),
        alt.Tooltip(f"{x_col}:Q", format=".1f"),
    ] + (tooltip_extras or [])
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=color)
        .encode(
            y=alt.Y(f"{y_col}:N", sort=None,
                    axis=alt.Axis(title=None, labelFontSize=10, labelLimit=160)),
            x=alt.X(f"{x_col}:Q",
                    scale=alt.Scale(domain=[0, x_max]),
                    axis=alt.Axis(title=x_title)),
            tooltip=tips,
        )
        .properties(height=height)
    )


def _stacked_bar(
    df: pd.DataFrame,
    id_col: str, value_cols: list[str], labels: list[str],
    colors: list[str], x_title: str = "Count", height: int = 480,
    limit: int = 20,
):
    df_sub  = df[[id_col] + value_cols].head(limit).copy()
    df_melt = df_sub.melt(id_vars=id_col, var_name="_raw", value_name="Count")
    label_map = dict(zip(value_cols, labels))
    df_melt["Category"] = df_melt["_raw"].map(label_map)
    return (
        alt.Chart(df_melt)
        .mark_bar()
        .encode(
            y=alt.Y(f"{id_col}:N", sort=None,
                    axis=alt.Axis(title=None, labelFontSize=10, labelLimit=140)),
            x=alt.X("Count:Q", axis=alt.Axis(title=x_title)),
            color=alt.Color("Category:N",
                            scale=alt.Scale(domain=labels, range=colors),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[f"{id_col}:N", "Category:N", "Count:Q"],
        )
        .properties(height=height)
    )


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RESTAURANT CARDS (fetched from /domain/restaurants/<nh>)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_individual_restaurants(hood_filter: str, api_base: str) -> list[dict]:
    """
    Tries to fetch individual restaurant records from the FastAPI backend.
    Returns an empty list silently if endpoint is unavailable.
    """
    import requests
    import math

    if not hood_filter or not api_base:
        return []
    try:
        hood_enc = hood_filter.replace(" ", "%20")
        r = requests.get(
            f"{api_base}/overview/domain/restaurants/individual",
            params={"neighborhood": hood_filter},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            rests = data.get("restaurants", [])
        else:
            return []
    except Exception:
        return []

    # score and sort
    scored = []
    for rest in rests:
        rating  = float(rest.get("rating", 0) or 0)
        reviews = int(rest.get("reviews", 0) or 0)
        if rating > 0 and reviews > 0:
            rest["_score"] = rating * math.log(reviews + 1)
            scored.append(rest)
    scored.sort(key=lambda x: -x["_score"])
    return scored[:18]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SVG DONUT CHART HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _svg_donut(slices, total, center_label="", center_sub="", size=320):
    """
    Pure HTML/SVG donut with:
      - Arc segments sized by value
      - % + count labels on each segment (outside ring, with leader line)
      - Center annotation (big number + sub-label)
      - Legend row below
    slices : list of (label, value, hex_color)
    """
    import math

    if total <= 0:
        return "<p style='color:rgba(255,255,255,0.3);font-size:12px;'>No data</p>"

    cx = cy = size / 2
    r_outer = size * 0.38
    r_inner = size * 0.22
    r_label = size * 0.47

    def polar(r, angle):
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def arc_path(r_out, r_in, a0, a1):
        ox1, oy1 = polar(r_out, a0)
        ox2, oy2 = polar(r_out, a1)
        ix1, iy1 = polar(r_in,  a1)
        ix2, iy2 = polar(r_in,  a0)
        lg = 1 if (a1 - a0) > math.pi else 0
        return (
            "M %.3f %.3f " % (ox1, oy1) +
            "A %.3f %.3f 0 %d 1 %.3f %.3f " % (r_out, r_out, lg, ox2, oy2) +
            "L %.3f %.3f " % (ix1, iy1) +
            "A %.3f %.3f 0 %d 0 %.3f %.3f " % (r_in, r_in, lg, ix2, iy2) +
            "Z"
        )

    paths  = []
    labels = []
    gap    = 0.025
    angle  = -math.pi / 2

    for label, value, color in slices:
        if value <= 0:
            continue
        frac   = value / total
        sweep  = frac * 2 * math.pi - gap
        end_a  = angle + sweep
        mid_a  = angle + sweep / 2

        paths.append(
            '<path d="%s" fill="%s" opacity="0.92"/>' % (arc_path(r_outer, r_inner, angle, end_a), color)
        )

        if frac > 0.04:
            lx, ly    = polar(r_label, mid_a)
            ex, ey    = polar(r_outer + 4, mid_a)
            lax, lay  = polar(r_label - 8, mid_a)
            pct_str   = "%.1f%%" % (frac * 100)
            cnt_str   = "(%d)" % int(value)
            labels.append(
                '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.2" opacity="0.55"/>' % (ex, ey, lax, lay, color) +
                '<text x="%.1f" y="%.1f" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="700" fill="%s">%s</text>' % (lx, ly - 7, color, pct_str) +
                '<text x="%.1f" y="%.1f" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="rgba(255,255,255,0.45)">%s</text>' % (lx, ly + 7, cnt_str)
            )

        angle = end_a + gap

    center = (
        '<text x="%.1f" y="%.1f" text-anchor="middle" dominant-baseline="middle" font-size="26" font-weight="700" fill="#e2e8f0">%s</text>' % (cx, cy - 10, center_label) +
        '<text x="%.1f" y="%.1f" text-anchor="middle" dominant-baseline="middle" font-size="11" fill="rgba(255,255,255,0.38)">%s</text>' % (cx, cy + 16, center_sub)
    )

    legend_items = []
    for label, value, color in slices:
        if value <= 0:
            continue
        pct = value / total * 100
        legend_items.append(
            '<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:11px;color:rgba(255,255,255,0.65);">'
            '<span style="width:10px;height:10px;border-radius:2px;background:%s;display:inline-block;"></span>'
            '%s <span style="color:%s;font-weight:700;">%.1f%%</span>'
            '</span>' % (color, label, color, pct)
        )

    return (
        '<div style="display:flex;flex-direction:column;align-items:center;">'
        '<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" style="overflow:visible;">'
        '%s%s%s'
        '</svg>'
        '<div style="display:flex;flex-wrap:wrap;justify-content:center;margin-top:10px;gap:4px;">%s</div>'
        '</div>'
    ) % (size, size, size, size, "".join(paths), "".join(labels), center, "".join(legend_items))


def render_restaurant_deep_dive(
    domain_data: dict,
    hood_filter: Optional[str] = None,
    api_base: str = "http://localhost:8001",
):
    """
    Renders the full Restaurant domain deep-dive inside Streamlit.

    Parameters
    ----------
    domain_data : dict
        Response from GET /overview/domain/restaurants[?neighborhood=...].
        Expected keys: neighborhoods (list), summary (dict).
    hood_filter : str | None
        Currently selected neighborhood, or None / "ALL" for citywide view.
    api_base : str
        FastAPI base URL — used to attempt individual restaurant fetch.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    neighborhoods: list[dict] = domain_data.get("neighborhoods", [])
    summary: dict              = domain_data.get("summary", {})

    if not neighborhoods:
        st.warning("No restaurant data available.")
        return

    df = pd.DataFrame(neighborhoods)

    # ── Normalise column names to lower snake_case ─────────────────────────
    df.columns = [c.lower() for c in df.columns]

    # Map every possible API field name variant → internal name used below.
    # Order matters: first match wins when both exist.
    _col_alias = [
        # score / grade
        ("restaurant_score",       "score"),
        ("restaurant_grade",       "grade"),
        # totals
        ("total_restaurants",      "total"),
        ("restaurants_per_sqmile", "per_sqmile"),
        # rating stats
        ("avg_rating",             "avg_rating"),
        ("total_reviews",          "total_reviews"),
        ("pct_high_quality",       "pct_high_quality"),
        # delivery
        ("pct_delivery",           "pct_delivery"),
        ("delivery_count",         "delivery_count"),
        # price tiers — both bare and _count variants
        ("budget_count",           "budget"),
        ("mid_range_count",        "mid_range"),
        ("upscale_count",          "upscale"),
        # rating tiers
        ("excellent_count",        "excellent"),
        ("good_count",             "good"),
        ("average_count",          "average"),
        ("poor_count",             "poor"),
        # cuisine diversity
        ("cuisine_diversity",      "cuisine_diversity"),
        # cuisine counts — all _count and bare variants
        ("ethnic_count",           "ethnic"),
        ("pizza_count",            "pizza"),
        ("cafe_bakery_count",      "cafe_bakery"),
        ("cafes_bakeries",         "cafe_bakery"),
        ("fast_food_count",        "fast_food"),
        ("breakfast_count",        "breakfast"),
        ("sandwiches_deli_count",  "sandwiches"),
        ("sandwiches_deli",        "sandwiches"),
        ("healthy_count",          "healthy"),
        ("american_count",         "american"),
        ("bar_count",              "bar"),
        ("other_count",            "other"),
        ("other",                  "other"),
        # description
        ("restaurant_description", "description"),
    ]
    for api_col, internal in _col_alias:
        if api_col in df.columns and internal not in df.columns:
            df[internal] = df[api_col]

    # ensure numeric for every column we will reference
    _numeric_cols = [
        "score", "total", "per_sqmile", "avg_rating", "total_reviews",
        "pct_high_quality", "pct_delivery", "delivery_count",
        "budget", "mid_range", "upscale",
        "excellent", "good", "average", "poor",
        "cuisine_diversity",
        "ethnic", "pizza", "cafe_bakery", "fast_food", "breakfast",
        "sandwiches", "healthy", "american", "bar", "other",
    ]
    for col in _numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE A: SINGLE NEIGHBORHOOD
    # ─────────────────────────────────────────────────────────────────────────
    if hood_filter and hood_filter.strip().upper() not in ("", "ALL"):
        row = df[df["neighborhood"].str.upper() == hood_filter.upper()]
        if row.empty:
            row = df.head(1)
        n = row.iloc[0]

        grade       = str(n.get("grade", "—")).strip()
        score       = float(n.get("score", 0))
        total       = int(n.get("total", 0))
        per_sqmile  = float(n.get("per_sqmile", 0))
        avg_rating  = float(n.get("avg_rating", 0))
        total_rev   = int(n.get("total_reviews", 0))
        pct_quality = float(n.get("pct_high_quality", 0))
        pct_del     = float(n.get("pct_delivery", 0))
        budget      = int(n.get("budget", 0))
        mid_range   = int(n.get("mid_range", 0))
        upscale     = int(n.get("upscale", 0))
        def _safe_int(series, *keys):
            for k in keys:
                try:
                    v = series[k]
                    if v is not None and str(v) not in ("nan", ""):
                        return int(float(v))
                except (KeyError, TypeError, ValueError):
                    pass
            return 0

        exc_ct  = _safe_int(n, "excellent", "excellent_count")
        good_ct = _safe_int(n, "good",      "good_count")
        avg_ct  = _safe_int(n, "average",   "average_count")
        poor_ct = _safe_int(n, "poor",      "poor_count")
        # If explicit counts still zero, derive approximate split from pct_high_quality
        if exc_ct + good_ct + avg_ct + poor_ct == 0 and total > 0:
            hq_frac  = min(float(n.get("pct_high_quality", 0) if hasattr(n, "get") else 0), 100) / 100
            exc_frac = max(0.0, hq_frac - 0.15)
            good_ct  = int(total * max(0.0, hq_frac - exc_frac))
            exc_ct   = int(total * exc_frac)
            poor_ct  = max(0, int(total * 0.08))
            avg_ct   = max(0, total - exc_ct - good_ct - poor_ct)
        description = str(n.get("description", "")) if n.get("description") else None

        # ── 4 KPI tiles ────────────────────────────────────────────────────
        bg, fg = GRADE_BG.get(grade, ("rgba(100,116,139,0.2)", "#94a3b8"))
        grade_badge = (
            f'<span style="background:{bg};color:{fg};border:1px solid {fg}44;'
            f'padding:3px 12px;border-radius:999px;font-size:11px;font-weight:700;">'
            f'{grade}</span>'
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="rdd-kpi accent">'
                f'<div class="rdd-kpi-label">Restaurant Score</div>'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span class="rdd-kpi-val">{score:.0f}/100</span>'
                f'{grade_badge}</div>'
                f'<div class="rdd-kpi-sub">{total:,} venues · {per_sqmile:.0f}/mi²</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                _kpi("Avg Rating",
                     f"{avg_rating:.2f} / 5",
                     f"{total_rev:,} total reviews"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                _kpi("High Quality",
                     f"{pct_quality:.1f}%",
                     "% Excellent or Good rated"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                _kpi("Delivery Access",
                     f"{pct_del:.1f}%",
                     f"{int(n.get('delivery_count', 0)):,} restaurants"),
                unsafe_allow_html=True,
            )

        # ── AI Narrative ───────────────────────────────────────────────────
        if description:
            st.markdown(
                f'<div class="rdd-narrative">'
                f'<div class="rdd-narrative-title">🍽 Dining Intelligence — {hood_filter}</div>'
                f'{description}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Row 1: Rating donut  +  Price mix donut ────────────────────────
        col_rd, col_pd = st.columns(2, gap="medium")

        with col_rd:
            st.markdown(_section_open("Rating Distribution", f"{hood_filter} · by quality tier"),
                        unsafe_allow_html=True)
            if exc_ct + good_ct + avg_ct + poor_ct > 0:
                st.markdown(
                    _svg_donut(
                        slices=[
                            ("Excellent", exc_ct,  "#22c55e"),
                            ("Good",      good_ct, "#3b82f6"),
                            ("Average",   avg_ct,  "#f59e0b"),
                            ("Poor",      poor_ct, "#ef4444"),
                        ],
                        total=exc_ct + good_ct + avg_ct + poor_ct,
                        center_label=str(exc_ct + good_ct + avg_ct + poor_ct),
                        center_sub="restaurants",
                        size=340,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info("Rating breakdown not available.")
            st.markdown(_section_close(), unsafe_allow_html=True)

        with col_pd:
            st.markdown(_section_open("Price Range Mix", f"{hood_filter} · budget to upscale"),
                        unsafe_allow_html=True)
            if budget + mid_range + upscale > 0:
                st.markdown(
                    _svg_donut(
                        slices=[
                            ("Budget ($)",     budget,    "#14b8a6"),
                            ("Mid-range ($$)", mid_range, "#fb923c"),
                            ("Upscale ($$$+)", upscale,   "#ec4899"),
                        ],
                        total=budget + mid_range + upscale,
                        center_label=str(budget + mid_range + upscale),
                        center_sub="restaurants",
                        size=340,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info("Price range data not available.")
            st.markdown(_section_close(), unsafe_allow_html=True)

        # ── Row 2: Cuisine bar  +  Quality radar ──────────────────────────
        col_cb, col_qr = st.columns(2, gap="medium")

        with col_cb:
            st.markdown(_section_open("Cuisine Categories", f"{hood_filter} · by count"),
                        unsafe_allow_html=True)
            def _cv(series, *keys):
                """Safe int from a pandas Series — tries each key, returns 0 if all missing."""
                for k in keys:
                    try:
                        v = series[k]
                        if v is not None and str(v) not in ("nan", ""):
                            return int(float(v))
                    except (KeyError, TypeError, ValueError):
                        pass
                return 0

            cuisine_map_raw = {
                "Ethnic":      _cv(n, "ethnic",    "ethnic_count"),
                "Café/Bakery": _cv(n, "cafe_bakery","cafe_bakery_count","cafes_bakeries"),
                "Fast Food":   _cv(n, "fast_food",  "fast_food_count"),
                "Healthy":     _cv(n, "healthy",    "healthy_count"),
                "American":    _cv(n, "american",   "american_count"),
                "Pizza":       _cv(n, "pizza",      "pizza_count"),
                "Breakfast":   _cv(n, "breakfast",  "breakfast_count"),
                "Sandwiches":  _cv(n, "sandwiches", "sandwiches_deli_count","sandwiches_deli"),
                "Bar":         _cv(n, "bar",        "bar_count"),
                "Other":       _cv(n, "other",      "other_count"),
            }
            # If every specific category is 0 but total > 0, the API isn't
            # returning the breakdown yet — show a single "All restaurants" bar
            # so the chart is never empty.
            known_total = sum(cuisine_map_raw.values())
            if known_total == 0 and total > 0:
                cuisine_map_raw["All restaurants"] = total
            cuisine_map = {k: v for k, v in cuisine_map_raw.items() if v > 0}
            if cuisine_map:
                df_cu = pd.DataFrame(
                    sorted(cuisine_map.items(), key=lambda x: -x[1]),
                    columns=["Cuisine", "Count"],
                )
                bar_cu = (
                    alt.Chart(df_cu)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y("Cuisine:N", sort=None,
                                axis=alt.Axis(title=None, labelFontSize=11, labelLimit=120)),
                        x=alt.X("Count:Q", axis=alt.Axis(title="Restaurants")),
                        color=alt.Color("Cuisine:N",
                                        scale=alt.Scale(range=CUISINE_COLORS),
                                        legend=None),
                        tooltip=["Cuisine:N", "Count:Q"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(bar_cu, use_container_width=True)
            else:
                st.info("Cuisine breakdown not available.")
            st.markdown(_section_close(), unsafe_allow_html=True)

        with col_qr:
            st.markdown(_section_open("Quality Radar", f"{hood_filter} · normalised 0–100"),
                        unsafe_allow_html=True)

            # Five axes, each normalised to 0–100
            diversity_score = min(float(n.get("cuisine_diversity", 0)) / 8 * 100, 100)
            density_score   = min(per_sqmile / 4000 * 100, 100)

            radar_data = {
                "Avg Rating":      round(avg_rating / 5 * 100, 1),
                "High Quality":    round(pct_quality, 1),
                "Delivery Cover":  round(pct_del, 1),
                "Cuisine Variety": round(diversity_score, 1),
                "Density":         round(density_score, 1),
            }
            df_radar = pd.DataFrame(
                [{"Metric": k, "Score": v} for k, v in radar_data.items()]
            )
            # Altair doesn't natively do polar charts — use a stacked bar as an
            # approximation that matches the app's design language
            bar_radar = (
                alt.Chart(df_radar)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#fb923c")
                .encode(
                    y=alt.Y("Metric:N", sort=None,
                            axis=alt.Axis(title=None, labelFontSize=11, labelLimit=130)),
                    x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 100]),
                            axis=alt.Axis(title="Score (normalised 0–100)")),
                    tooltip=["Metric:N", alt.Tooltip("Score:Q", format=".1f")],
                )
                .properties(height=240)
            )
            labels_radar = (
                alt.Chart(df_radar)
                .mark_text(align="left", dx=4, fontSize=11,
                           fontWeight="bold", color="#e2e8f0")
                .encode(
                    y=alt.Y("Metric:N", sort=None),
                    x=alt.X("Score:Q"),
                    text=alt.Text("Score:Q", format=".0f"),
                )
            )
            st.altair_chart(
                alt.layer(bar_radar, labels_radar).properties(height=240),
                use_container_width=True,
            )
            st.markdown(_section_close(), unsafe_allow_html=True)

        # ── Individual restaurant cards ────────────────────────────────────
        api_base_url = api_base
        try:
            api_base_url = st.session_state.get(
                "_api_base",
                __import__("os").getenv("API_BASE_URL", api_base),
            )
        except Exception:
            pass

        individual = _fetch_individual_restaurants(hood_filter, api_base_url)

        if individual:
            st.markdown(
                _section_open(
                    f"Top Restaurants — {hood_filter}",
                    "Ranked by rating × review volume",
                ),
                unsafe_allow_html=True,
            )
            cols_per_row = 3
            for i in range(0, len(individual), cols_per_row):
                row_cols = st.columns(cols_per_row, gap="small")
                for j, col in enumerate(row_cols):
                    idx = i + j
                    if idx < len(individual):
                        col.markdown(_rest_card(individual[idx]), unsafe_allow_html=True)
            st.markdown(_section_close(), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MODE B: ALL NEIGHBORHOODS (citywide)
    # ─────────────────────────────────────────────────────────────────────────
    else:
        total_rest   = int(summary.get("total_restaurants_citywide", df["total"].sum() if "total" in df.columns else 0))
        avg_rating   = float(summary.get("avg_rating_citywide", df["avg_rating"].mean() if "avg_rating" in df.columns else 0))
        n_excellent  = int((df["grade"] == "EXCELLENT").sum()) if "grade" in df.columns else 0
        best_row     = df.loc[df["score"].idxmax()] if "score" in df.columns and len(df) > 0 else None

        # ── Summary ribbon ──────────────────────────────────────────────────
        best_name  = best_row["neighborhood"] if best_row is not None else "—"
        best_score = float(best_row["score"]) if best_row is not None else 0
        top_quality_row = df.loc[df["pct_high_quality"].idxmax()] if "pct_high_quality" in df.columns and len(df) > 0 else None
        top_quality_name = top_quality_row["neighborhood"] if top_quality_row is not None else "—"
        top_quality_pct  = float(top_quality_row["pct_high_quality"]) if top_quality_row is not None else 0

        st.markdown(
            f'<div class="rdd-ribbon">'
            f'🍽 Greater Boston spans <b>{total_rest:,} restaurants</b> across '
            f'<b>{len(df)} neighborhoods</b> with a citywide avg rating of '
            f'<b>{avg_rating:.2f}/5</b>. &nbsp;·&nbsp; '
            f'<b>{n_excellent}</b> neighborhoods earned an EXCELLENT grade. &nbsp;·&nbsp; '
            f'<b>{best_name}</b> leads with a score of <b>{best_score:.0f}/100</b>. &nbsp;·&nbsp; '
            f'Highest quality: <b>{top_quality_name}</b> at <b>{top_quality_pct:.1f}%</b> high-quality.'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── KPI cards (citywide) ───────────────────────────────────────────
        avg_score   = float(df["score"].mean()) if "score" in df.columns else 0
        avg_pct_del = float(df["pct_delivery"].mean()) if "pct_delivery" in df.columns else 0
        top_del_row = df.loc[df["pct_delivery"].idxmax()] if "pct_delivery" in df.columns and len(df) > 0 else None
        top_del_name = top_del_row["neighborhood"] if top_del_row is not None else "—"
        top_del_pct  = float(top_del_row["pct_delivery"]) if top_del_row is not None else 0

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            st.markdown(
                _kpi("Total Restaurants", f"{total_rest:,}",
                     f"Across {len(df)} neighborhoods", accent=True),
                unsafe_allow_html=True,
            )
        with kc2:
            st.markdown(
                _kpi("Citywide Avg Rating", f"{avg_rating:.2f} / 5",
                     f"Avg score {avg_score:.0f}/100"),
                unsafe_allow_html=True,
            )
        with kc3:
            st.markdown(
                _kpi("Best Neighborhood", best_name,
                     f"Score: {best_score:.0f}/100 · EXCELLENT grade"),
                unsafe_allow_html=True,
            )
        with kc4:
            st.markdown(
                _kpi("Best Delivery Coverage", top_del_name,
                     f"{top_del_pct:.1f}% · citywide avg {avg_pct_del:.1f}%"),
                unsafe_allow_html=True,
            )

        # ── Row 1: Score chart  +  Price mix ──────────────────────────────
        df_sorted = df.sort_values("score", ascending=False)
        col_sc, col_pm = st.columns(2, gap="medium")

        with col_sc:
            st.markdown(_section_open("Restaurant Score — Top 20", "Ranked by composite score"),
                        unsafe_allow_html=True)
            chart_sc = _bar_chart(
                df_sorted.head(20), "score", "neighborhood",
                "Restaurant Score", color="#fb923c",
                x_max=100, height=max(20 * 28 + 60, 400),
                tooltip_extras=[
                    alt.Tooltip("grade:N", title="Grade"),
                    alt.Tooltip("total:Q", title="Total Restaurants"),
                    alt.Tooltip("avg_rating:Q", format=".2f", title="Avg Rating"),
                ],
            )
            st.altair_chart(chart_sc, use_container_width=True)
            st.markdown(_section_close(), unsafe_allow_html=True)

        with col_pm:
            st.markdown(_section_open("Price Range Mix — Top 20", "Budget / mid-range / upscale split"),
                        unsafe_allow_html=True)
            stk = _stacked_bar(
                df_sorted.head(20),
                id_col="neighborhood",
                value_cols=["budget", "mid_range", "upscale"],
                labels=["Budget ($)", "Mid-range ($$)", "Upscale ($$$+)"],
                colors=PRICE_COLORS,
                x_title="Restaurant Count",
                height=max(20 * 28 + 60, 400),
            )
            st.altair_chart(stk, use_container_width=True)
            st.markdown(_section_close(), unsafe_allow_html=True)

        # ── Row 2: Quality scatter  +  Delivery vs Quality ────────────────
        col_qs, col_dq = st.columns(2, gap="medium")

        with col_qs:
            st.markdown(_section_open("Avg Rating vs High-Quality %",
                                      "Bubble size = total restaurants"),
                        unsafe_allow_html=True)
            df_scatter = df.dropna(subset=["avg_rating", "pct_high_quality", "total"]).copy()
            if not df_scatter.empty:
                scatter = (
                    alt.Chart(df_scatter)
                    .mark_circle(opacity=0.75)
                    .encode(
                        x=alt.X("avg_rating:Q",
                                scale=alt.Scale(zero=False),
                                axis=alt.Axis(title="Avg Rating")),
                        y=alt.Y("pct_high_quality:Q",
                                axis=alt.Axis(title="High Quality %")),
                        size=alt.Size("total:Q",
                                      scale=alt.Scale(range=[40, 600]),
                                      legend=None),
                        color=alt.Color("grade:N",
                                        scale=alt.Scale(
                                            domain=["EXCELLENT", "GOOD", "MODERATE", "LIMITED"],
                                            range=[GRADE_COLORS[g] for g in
                                                   ["EXCELLENT", "GOOD", "MODERATE", "LIMITED"]]),
                                        legend=alt.Legend(title="Grade", orient="bottom",
                                                          direction="horizontal", labelFontSize=10)),
                        tooltip=[
                            alt.Tooltip("neighborhood:N"),
                            alt.Tooltip("avg_rating:Q", format=".2f", title="Avg Rating"),
                            alt.Tooltip("pct_high_quality:Q", format=".1f", title="High Quality %"),
                            alt.Tooltip("total:Q", title="Total Restaurants"),
                            alt.Tooltip("grade:N", title="Grade"),
                        ],
                    )
                    .properties(height=340)
                )
                st.altair_chart(scatter, use_container_width=True)
            else:
                st.info("Scatter data not available.")
            st.markdown(_section_close(), unsafe_allow_html=True)

        with col_dq:
            st.markdown(_section_open("Delivery Coverage — Top 20",
                                      "% of restaurants offering delivery"),
                        unsafe_allow_html=True)
            df_del = df.sort_values("pct_delivery", ascending=False).head(20)
            bar_del = _bar_chart(
                df_del, "pct_delivery", "neighborhood",
                "% With Delivery", color="#14b8a6",
                x_max=100,
                height=max(20 * 28 + 60, 340),
                tooltip_extras=[
                    alt.Tooltip("delivery_count:Q", title="Delivery Venues"),
                    alt.Tooltip("total:Q", title="Total Restaurants"),
                ],
            )
            st.altair_chart(bar_del, use_container_width=True)
            st.markdown(_section_close(), unsafe_allow_html=True)

        # ── Row 3: Cuisine diversity heatmap ──────────────────────────────
        st.markdown(_section_open("Cuisine Diversity — All Neighborhoods",
                                  "Restaurant count by category per neighborhood"),
                    unsafe_allow_html=True)

        cuisine_cols_available = [
            c for c in ["ethnic", "pizza", "cafe_bakery", "fast_food", "breakfast",
                        "sandwiches", "healthy", "american", "bar", "other"]
            if c in df.columns
        ]
        if cuisine_cols_available:
            label_map = {
                "ethnic": "Ethnic", "pizza": "Pizza",
                "cafe_bakery": "Café/Bakery", "fast_food": "Fast Food",
                "breakfast": "Breakfast", "sandwiches": "Sandwiches",
                "healthy": "Healthy", "american": "American",
                "bar": "Bar", "other": "Other",
            }
            # Sort by score on df FIRST, then extract cuisine cols — avoids KeyError on df_heat
            sort_col = "score" if "score" in df.columns else cuisine_cols_available[0]
            df_ordered = df.sort_values(sort_col, ascending=False).head(25)
            df_heat = df_ordered[["neighborhood"] + cuisine_cols_available].copy()
            df_melt_h = df_heat.melt(
                id_vars="neighborhood",
                var_name="_raw", value_name="Count",
            )
            df_melt_h["Cuisine"] = df_melt_h["_raw"].map(label_map).fillna(df_melt_h["_raw"])

            heatmap = (
                alt.Chart(df_melt_h)
                .mark_rect(stroke="rgba(0,0,0,0.2)", strokeWidth=0.5)
                .encode(
                    x=alt.X("Cuisine:N",
                            axis=alt.Axis(title=None, labelAngle=-30, labelFontSize=11)),
                    y=alt.Y("neighborhood:N",
                            sort=alt.EncodingSortField("Count", order="descending"),
                            axis=alt.Axis(title=None, labelFontSize=10, labelLimit=160)),
                    color=alt.Color("Count:Q",
                                    scale=alt.Scale(domain=[0, None],
                                                    range=["#1e2d40", "#fb923c"]),
                                    legend=alt.Legend(title="Count", orient="right")),
                    tooltip=[
                        alt.Tooltip("neighborhood:N"),
                        alt.Tooltip("Cuisine:N"),
                        alt.Tooltip("Count:Q"),
                    ],
                )
                .properties(height=560)
            )
            txt_h = (
                alt.Chart(df_melt_h)
                .mark_text(fontSize=10, fontWeight="bold")
                .encode(
                    x=alt.X("Cuisine:N"),
                    y=alt.Y("neighborhood:N",
                            sort=alt.EncodingSortField("Count", order="descending")),
                    text=alt.Text("Count:Q", format=".0f"),
                    color=alt.condition(
                        alt.datum.Count > 50,
                        alt.value("#0f172a"),
                        alt.value("#e2e8f0"),
                    ),
                )
            )
            st.altair_chart(alt.layer(heatmap, txt_h).properties(height=560),
                            use_container_width=True)
        st.markdown(_section_close(), unsafe_allow_html=True)

        # ── Full ranked table ──────────────────────────────────────────────
        st.markdown(_section_open("All Neighborhoods Ranked",
                                  "Click a neighborhood in the sidebar to drill in"),
                    unsafe_allow_html=True)

        table_rows = ""
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            g     = str(row.get("grade", "—")).strip()
            bg, fg = GRADE_BG.get(g, ("rgba(100,116,139,0.15)", "#94a3b8"))
            score = float(row.get("score", 0))
            table_rows += (
                f"<tr>"
                f'<td class="rdd-rank-num">{i}</td>'
                f'<td style="color:#e2e8f0;font-weight:500;">{row.get("neighborhood","—")}</td>'
                f'<td style="color:rgba(255,255,255,0.38);font-size:11px;">{row.get("city","—")}</td>'
                f'<td><span style="background:{bg};color:{fg};border:1px solid {fg}44;'
                f'padding:2px 9px;border-radius:999px;font-size:10px;font-weight:700;">{g}</span></td>'
                f'<td style="color:rgba(255,255,255,0.55);">{int(row.get("total",0)):,}</td>'
                f'<td style="color:rgba(255,255,255,0.55);">{float(row.get("avg_rating",0)):.2f}</td>'
                f'<td style="color:rgba(255,255,255,0.55);">{float(row.get("pct_high_quality",0)):.1f}%</td>'
                f'<td>'
                f'<div class="rdd-score-bar-wrap">'
                f'<div class="rdd-score-bar"><div class="rdd-score-fill" style="width:{score}%;"></div></div>'
                f'<span class="rdd-score-val">{score:.0f}</span>'
                f'</div></td>'
                f"</tr>"
            )

        st.markdown(
            f'<table class="rdd-rank-table">'
            f'<thead><tr>'
            f'<th>#</th><th>Neighborhood</th><th>City</th><th>Grade</th>'
            f'<th>Venues</th><th>Avg ★</th><th>High Quality</th><th>Score</th>'
            f'</tr></thead>'
            f'<tbody>{table_rows}</tbody>'
            f'</table>',
            unsafe_allow_html=True,
        )
        st.markdown(_section_close(), unsafe_allow_html=True)