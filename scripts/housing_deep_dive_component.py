"""
housing_deep_dive_component.py — NeighbourWise AI
═══════════════════════════════════════════════════════════════════════════════
Standalone Housing domain deep-dive component.

Visualizations:
  Row 0 — Hero KPI ribbon (5 city-wide stats)
  Row 1 — [L] Affordability score bar  [R] Grade donut + Rent vs Score scatter
  Row 2 — Price per sqft ranked bar (colour-coded by group)
  Row 3 — [L] Living area vs bedrooms bubble  [R] Property age histogram
  Row 4 — Amenity radar / stacked bar (AC · Parking · Fireplace) top-15
  Row 5 — Condition quality heatmap (Excellent → Fair breakdown, Boston neighbourhoods)
  Row 6 — Property density (properties per sq mile) bar

Call:
    from housing_deep_dive_component import render_housing_deep_dive
    render_housing_deep_dive(domain_data, hood_filter)

`domain_data` is the dict returned by /overview/domain/housing.
Expected keys: neighborhoods (list of dicts matching the CSV columns, snake_cased).
`hood_filter` is either None (city-wide) or a neighbourhood name string.
"""

import streamlit as st
import pandas as pd
import altair as alt


# ── Grade palette ─────────────────────────────────────────────────────────────
GRADE_COLORS = {
    "AFFORDABLE":    "#22c55e",
    "AVERAGE":       "#f59e0b",
    "PREMIUM":       "#ef4444",
    "BELOW_AVERAGE": "#94a3b8",
}
GRADE_ORDER  = ["AFFORDABLE", "AVERAGE", "PREMIUM", "BELOW_AVERAGE"]

# ── Distribution-group accent colours ─────────────────────────────────────────
GROUP_COLORS = {
    "BOSTON":        "#60a5fa",
    "CAMBRIDGE":     "#a78bfa",
    "GREATER_BOSTON":"#34d399",
}

CONDITION_COLS   = ["excellent_count","very_good_count","good_count","average_count","fair_count"]
CONDITION_LABELS = ["Excellent","Very Good","Good","Average","Fair"]
CONDITION_COLORS = ["#22c55e","#86efac","#fbbf24","#f97316","#ef4444"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="section-card">',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def _end_section():
    st.markdown('</div>', unsafe_allow_html=True)


def _narrative(html: str, colour: str = "green"):
    border = "#10B981" if colour == "green" else "#60a5fa"
    bg     = "rgba(16,185,129,0.08)" if colour == "green" else "rgba(96,165,250,0.08)"
    border_col = "rgba(16,185,129,0.2)" if colour == "green" else "rgba(96,165,250,0.2)"
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border_col};'
        f'border-left:4px solid {border};padding:0.75rem 1rem;border-radius:10px;'
        f'margin-bottom:0.8rem;font-size:0.86rem;line-height:1.6;color:#e2e8f0;">'
        f'{html}</div>',
        unsafe_allow_html=True,
    )


def _kpi_ribbon(items):
    """5-column KPI ribbon matching the app's metric-card CSS."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names, then apply API→CSV field aliases so both
    the FastAPI response shape and the raw CSV shape work identically."""
    df.columns = [c.lower() for c in df.columns]

    # API field name  →  internal canonical name (CSV column name, lowercased)
    ALIASES = {
        # rent
        "avg_monthly_rent":       "avg_estimated_rent",
        # neighbourhood name — API uses 'neighborhood', CSV has 'neighborhood_name'
        "neighborhood":           "neighborhood_name",
        # score/grade — API may use shorter keys
        "housing_score":          "housing_score",      # same, keep for safety
        "housing_grade":          "housing_grade",      # same
        "score":                  "housing_score",
        "grade":                  "housing_grade",
        # group — API may call it 'group' or 'dist_group'
        "group":                  "distribution_group",
        "dist_group":             "distribution_group",
        # size fields
        "sq_miles":               "sqmiles",
        "area_sqmiles":           "sqmiles",
        # condition counts (API may send as 'cnt_excellent' etc.)
        "cnt_excellent":          "excellent_count",
        "cnt_very_good":          "very_good_count",
        "cnt_good":               "good_count",
        "cnt_average":            "average_count",
        "cnt_fair":               "fair_count",
        # amenity pcts — API may omit pct_ prefix
        "has_ac_pct":             "pct_has_ac",
        "has_parking_pct":        "pct_has_parking",
        "ac_pct":                 "pct_has_ac",
        "parking_pct":            "pct_has_parking",
    }

    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})

    # If 'neighborhood_name' still missing but 'neighborhood' present, alias it
    if "neighborhood_name" not in df.columns and "neighborhood" in df.columns:
        df = df.rename(columns={"neighborhood": "neighborhood_name"})

    # If 'avg_estimated_rent' still missing but 'avg_monthly_rent' present
    if "avg_estimated_rent" not in df.columns and "avg_monthly_rent" in df.columns:
        df["avg_estimated_rent"] = df["avg_monthly_rent"]

    return df


def _coerce(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _col(df: pd.DataFrame, primary: str, *fallbacks) -> str:
    """Return the first column name that exists in df, or primary (missing → NaN series)."""
    for name in (primary,) + fallbacks:
        if name in df.columns:
            return name
    return primary  # will produce KeyError → caller gets NaN gracefully via .get()


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _chart_affordability_bar(df: pd.DataFrame) -> alt.Chart:
    """Horizontal bar — housing_score, coloured by grade."""
    dff = df[df["housing_score"].notna()].copy()
    dff = dff.sort_values("housing_score", ascending=False).head(30)
    rent_col  = _col(dff, "avg_estimated_rent", "avg_monthly_rent")
    name_col  = _col(dff, "neighborhood_name", "neighborhood")
    rent_tip  = alt.Tooltip(f"{rent_col}:Q", format=",.0f", title="Avg Rent $") if rent_col in dff.columns else alt.Tooltip("housing_score:Q", title="Score")
    bars = alt.Chart(dff).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
        y=alt.Y(f"{name_col}:N",
                sort=alt.EncodingSortField("housing_score", order="descending"),
                axis=alt.Axis(title=None, labelFontSize=10, labelLimit=170)),
        x=alt.X("housing_score:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="Affordability Score", grid=True)),
        color=alt.Color("housing_grade:N",
                        scale=alt.Scale(domain=GRADE_ORDER,
                                        range=[GRADE_COLORS[g] for g in GRADE_ORDER]),
                        legend=alt.Legend(title="Grade", orient="bottom")),
        tooltip=[f"{name_col}:N",
                 alt.Tooltip("housing_score:Q", format=".1f", title="Score"),
                 "housing_grade:N",
                 rent_tip,
                 alt.Tooltip("avg_price_per_sqft:Q",   format=".2f",  title="$/sqft"),
                 alt.Tooltip("avg_living_area_sqft:Q", format=",.0f", title="Avg sqft")],
    )
    labels = alt.Chart(dff).mark_text(align="left", dx=4, fontSize=10, color="#e2e8f0").encode(
        y=alt.Y(f"{name_col}:N",
                sort=alt.EncodingSortField("housing_score", order="descending")),
        x=alt.X("housing_score:Q"),
        text=alt.Text("housing_score:Q", format=".0f"),
    )
    return alt.layer(bars, labels).properties(height=max(360, len(dff) * 18))


def _chart_grade_donut(df: pd.DataFrame) -> alt.Chart:
    dff = df[df["housing_grade"].notna() & (df["housing_grade"] != "INSUFFICIENT DATA")]
    grade_counts = dff["housing_grade"].value_counts().reset_index()
    grade_counts.columns = ["Grade", "Count"]
    grade_counts["Pct"] = (grade_counts["Count"] / grade_counts["Count"].sum() * 100).round(0).astype(int).astype(str) + "%"
    donut = alt.Chart(grade_counts).mark_arc(innerRadius=55, outerRadius=100,
                                              stroke="#1a1a2e", strokeWidth=2).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color("Grade:N",
                        scale=alt.Scale(domain=GRADE_ORDER,
                                        range=[GRADE_COLORS[g] for g in GRADE_ORDER]),
                        legend=alt.Legend(title=None, orient="bottom",
                                          direction="horizontal", labelFontSize=11)),
        order=alt.Order("Count:Q", sort="descending"),
        tooltip=["Grade:N", "Count:Q", "Pct:N"],
    )
    labels = alt.Chart(grade_counts).mark_text(radius=118, fontSize=12, fontWeight="bold",
                                                color="#e2e8f0").encode(
        theta=alt.Theta("Count:Q", stack=True),
        order=alt.Order("Count:Q", sort="descending"),
        text=alt.Text("Pct:N"),
    )
    return alt.layer(donut, labels).properties(height=280)


def _chart_rent_vs_score(df: pd.DataFrame) -> alt.Chart:
    rent_col = _col(df, "avg_estimated_rent", "avg_monthly_rent")
    name_col = _col(df, "neighborhood_name", "neighborhood")
    if rent_col not in df.columns:
        return alt.Chart(pd.DataFrame()).mark_text().encode()
    dff = df[df[rent_col].notna() & df["housing_score"].notna()].copy()
    return alt.Chart(dff).mark_circle(size=90, opacity=0.85).encode(
        x=alt.X(f"{rent_col}:Q",
                axis=alt.Axis(title="Avg Monthly Rent ($)", format="$,.0f")),
        y=alt.Y("housing_score:Q",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(title="Affordability Score")),
        color=alt.Color("housing_grade:N",
                        scale=alt.Scale(domain=GRADE_ORDER,
                                        range=[GRADE_COLORS[g] for g in GRADE_ORDER]),
                        legend=None),
        tooltip=[f"{name_col}:N",
                 alt.Tooltip("housing_score:Q",       format=".1f",  title="Score"),
                 alt.Tooltip(f"{rent_col}:Q",         format=",.0f", title="Rent $"),
                 "housing_grade:N"],
    ).properties(height=220)


def _chart_price_per_sqft(df: pd.DataFrame) -> alt.Chart:
    """Full ranked bar — price per sqft, coloured by distribution group."""
    dff = df[df["avg_price_per_sqft"].notna()].copy()
    dff = dff.sort_values("avg_price_per_sqft", ascending=False)
    name_col = _col(dff, "neighborhood_name", "neighborhood")
    grp_col  = _col(dff, "distribution_group", "group", "dist_group")
    bars = alt.Chart(dff).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
        x=alt.X(f"{name_col}:N",
                sort=alt.EncodingSortField("avg_price_per_sqft", order="descending"),
                axis=alt.Axis(title=None, labelAngle=-40, labelFontSize=10, labelLimit=140)),
        y=alt.Y("avg_price_per_sqft:Q",
                axis=alt.Axis(title="Avg Price per Sqft ($)", grid=True, format="$,.0f")),
        color=alt.Color(f"{grp_col}:N",
                        scale=alt.Scale(
                            domain=list(GROUP_COLORS.keys()),
                            range=list(GROUP_COLORS.values())),
                        legend=alt.Legend(title="Group", orient="top-right")),
        tooltip=[f"{name_col}:N",
                 alt.Tooltip("avg_price_per_sqft:Q",  format=".2f",  title="$/sqft"),
                 alt.Tooltip("avg_assessed_value:Q",  format=",.0f", title="Avg Value $"),
                 f"{grp_col}:N",
                 "housing_grade:N"],
    ).properties(height=300)
    rule = alt.Chart(pd.DataFrame({"y": [dff["avg_price_per_sqft"].median()]})).mark_rule(
        color="#e2e8f0", strokeDash=[6, 3], strokeWidth=1.5,
    ).encode(y="y:Q")
    return alt.layer(bars, rule)


def _chart_living_area_bubble(df: pd.DataFrame) -> alt.Chart:
    """Bubble — avg_living_area_sqft (x) vs avg_bedrooms (y), size = total_properties."""
    dff = df[df["avg_living_area_sqft"].notna() & df["avg_bedrooms"].notna()].copy()
    name_col = _col(dff, "neighborhood_name", "neighborhood")
    return alt.Chart(dff).mark_circle(opacity=0.8).encode(
        x=alt.X("avg_living_area_sqft:Q",
                axis=alt.Axis(title="Avg Living Area (sqft)")),
        y=alt.Y("avg_bedrooms:Q",
                scale=alt.Scale(domain=[0, 7]),
                axis=alt.Axis(title="Avg Bedrooms")),
        size=alt.Size("total_properties:Q",
                      scale=alt.Scale(range=[60, 1200]),
                      legend=alt.Legend(title="# Properties")),
        color=alt.Color("housing_grade:N",
                        scale=alt.Scale(domain=GRADE_ORDER,
                                        range=[GRADE_COLORS[g] for g in GRADE_ORDER]),
                        legend=None),
        tooltip=[f"{name_col}:N",
                 alt.Tooltip("avg_living_area_sqft:Q", format=",.0f", title="Avg sqft"),
                 alt.Tooltip("avg_bedrooms:Q",         format=".1f",  title="Avg beds"),
                 alt.Tooltip("total_properties:Q",     format=",",    title="Properties"),
                 "housing_grade:N"],
    ).properties(height=340)


def _chart_property_age(df: pd.DataFrame) -> alt.Chart:
    """Horizontal bar — avg_property_age sorted descending (oldest first)."""
    dff = df[df["avg_property_age"].notna()].copy()
    dff = dff.sort_values("avg_property_age", ascending=False).head(30)
    name_col = _col(dff, "neighborhood_name", "neighborhood")
    grp_col  = _col(dff, "distribution_group", "group", "dist_group")
    bars = alt.Chart(dff).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y(f"{name_col}:N",
                sort=alt.EncodingSortField("avg_property_age", order="descending"),
                axis=alt.Axis(title=None, labelFontSize=10, labelLimit=160)),
        x=alt.X("avg_property_age:Q",
                axis=alt.Axis(title="Avg Property Age (years)")),
        color=alt.Color("avg_property_age:Q",
                        scale=alt.Scale(domain=[30, 125], range=["#67e8f9", "#b45309"]),
                        legend=alt.Legend(title="Age (yrs)")),
        tooltip=[f"{name_col}:N",
                 alt.Tooltip("avg_property_age:Q",  format=".0f", title="Avg Age"),
                 alt.Tooltip("oldest_year_built:Q", format="d",   title="Oldest Built"),
                 alt.Tooltip("newest_year_built:Q", format="d",   title="Newest Built"),
                 f"{grp_col}:N"],
    )
    labels = alt.Chart(dff).mark_text(align="left", dx=4, fontSize=9, color="#e2e8f0").encode(
        y=alt.Y(f"{name_col}:N",
                sort=alt.EncodingSortField("avg_property_age", order="descending")),
        x=alt.X("avg_property_age:Q"),
        text=alt.Text("avg_property_age:Q", format=".0f"),
    )
    return alt.layer(bars, labels).properties(height=max(320, len(dff) * 17))


def _chart_amenities(df: pd.DataFrame) -> alt.Chart:
    """Stacked % bar — AC / Parking / Fireplace for top-20 by amenity_rate."""
    dff = df[df["pct_has_ac"].notna() & df["pct_has_parking"].notna()].copy()
    dff = dff.nlargest(20, "amenity_rate") if "amenity_rate" in dff.columns else dff.head(20)
    name_col = _col(dff, "neighborhood_name", "neighborhood")

    dff["pct_has_fireplace"] = (
        dff["has_fireplace_count"].fillna(0) / dff["total_properties"].replace(0, pd.NA) * 100
    ).round(1) if "has_fireplace_count" in dff.columns else 0.0

    melt = dff[[name_col,"pct_has_ac","pct_has_parking","pct_has_fireplace"]].melt(
        id_vars=name_col, var_name="Amenity", value_name="Pct"
    )
    label_map = {
        "pct_has_ac":        "Air Conditioning",
        "pct_has_parking":   "Parking",
        "pct_has_fireplace": "Fireplace",
    }
    melt["Amenity"] = melt["Amenity"].map(label_map)

    return alt.Chart(melt).mark_bar().encode(
        y=alt.Y(f"{name_col}:N",
                sort=alt.EncodingSortField(field="Pct", op="sum", order="descending"),
                axis=alt.Axis(title=None, labelFontSize=10, labelLimit=160)),
        x=alt.X("Pct:Q", axis=alt.Axis(title="% of Properties", format=".0f")),
        color=alt.Color("Amenity:N",
                        scale=alt.Scale(
                            domain=["Air Conditioning","Parking","Fireplace"],
                            range=["#60a5fa","#34d399","#f97316"]),
                        legend=alt.Legend(title=None, orient="bottom",
                                          direction="horizontal")),
        tooltip=[f"{name_col}:N","Amenity:N",
                 alt.Tooltip("Pct:Q", format=".1f", title="% Properties")],
    ).properties(height=360)


def _chart_condition_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Heatmap — condition tier % per neighbourhood (Boston only for readability)."""
    grp_col  = _col(df, "distribution_group", "group", "dist_group")
    name_col = _col(df, "neighborhood_name", "neighborhood")
    dff = df[df["excellent_count"].notna() & (df["total_properties"] > 0)].copy()
    if grp_col in dff.columns:
        boston = dff[dff[grp_col] == "BOSTON"].copy()
        dff = boston if not boston.empty else dff

    for col, lbl in zip(CONDITION_COLS, CONDITION_LABELS):
        if col in dff.columns:
            dff[lbl] = (dff[col] / dff["total_properties"] * 100).round(1)
        else:
            dff[lbl] = 0.0

    melt = dff[[name_col] + CONDITION_LABELS].melt(
        id_vars=name_col, var_name="Condition", value_name="Pct"
    )
    return alt.Chart(melt).mark_rect(stroke="#1a1a2e", strokeWidth=1).encode(
        x=alt.X("Condition:N",
                sort=CONDITION_LABELS,
                axis=alt.Axis(title=None, labelAngle=0, labelFontSize=11,
                              labelFontWeight="bold")),
        y=alt.Y(f"{name_col}:N",
                sort=alt.EncodingSortField(name_col, order="ascending"),
                axis=alt.Axis(title=None, labelFontSize=10, labelLimit=160)),
        color=alt.Color("Pct:Q",
                        scale=alt.Scale(domain=[0, 80], range=["#1e3a5f", "#22c55e"]),
                        legend=alt.Legend(title="% of Stock")),
        tooltip=[f"{name_col}:N","Condition:N",
                 alt.Tooltip("Pct:Q", format=".1f", title="% Properties")],
    ).properties(height=max(300, len(dff) * 19))


def _chart_density(df: pd.DataFrame) -> alt.Chart:
    """Bar — properties per sq mile, log-scaled, coloured by group."""
    dff = df[df["properties_per_sqmile"].notna() & (df["properties_per_sqmile"] > 0)].copy()
    dff = dff.sort_values("properties_per_sqmile", ascending=False)
    name_col = _col(dff, "neighborhood_name", "neighborhood")
    grp_col  = _col(dff, "distribution_group", "group", "dist_group")
    sqmi_col = _col(dff, "sqmiles", "sq_miles", "area_sqmiles")
    bars = alt.Chart(dff).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        x=alt.X(f"{name_col}:N",
                sort=alt.EncodingSortField("properties_per_sqmile", order="descending"),
                axis=alt.Axis(title=None, labelAngle=-40, labelFontSize=10, labelLimit=130)),
        y=alt.Y("properties_per_sqmile:Q",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(title="Properties / sq mile (log)", grid=True, format=",")),
        color=alt.Color(f"{grp_col}:N",
                        scale=alt.Scale(domain=list(GROUP_COLORS.keys()),
                                        range=list(GROUP_COLORS.values())),
                        legend=None),
        tooltip=[f"{name_col}:N",
                 alt.Tooltip("properties_per_sqmile:Q", format=",.1f", title="Props/sqmi"),
                 alt.Tooltip("total_properties:Q",      format=",",    title="Total Props"),
                 alt.Tooltip(f"{sqmi_col}:Q",           format=".2f",  title="Sq Miles"),
                 f"{grp_col}:N"],
    ).properties(height=260)
    return bars


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_housing_deep_dive(domain_data: dict, hood_filter: str = None):
    """
    Render the full Housing deep-dive inside the NeighbourWise Overview tab.

    Parameters
    ----------
    domain_data : dict   — response from /overview/domain/housing
    hood_filter : str    — selected neighbourhood name, or None for city-wide
    """

    raw = domain_data.get("neighborhoods", [])
    if not raw:
        st.info("Housing data not available.")
        return

    df = pd.DataFrame(raw)
    df = _norm(df)
    numeric_cols = [
        "housing_score","avg_price_per_sqft","avg_assessed_value",
        "avg_estimated_rent","avg_monthly_rent",          # both aliases coerced
        "avg_living_area_sqft","avg_bedrooms","avg_full_baths","total_properties",
        "condo_count","rental_count","avg_property_age","oldest_year_built","newest_year_built",
        "properties_per_sqmile","pct_good_condition","pct_has_ac","pct_has_parking",
        "amenity_rate","avg_condition_score","has_fireplace_count","has_ac_count",
        "has_parking_count","sqmiles",
        "excellent_count","very_good_count","good_count","average_count","fair_count",
    ]
    df = _coerce(df, numeric_cols)

    # Filter out INSUFFICIENT DATA rows for charts that need scores
    df_scored = df[df["housing_grade"] != "INSUFFICIENT DATA"].copy()

    # Single neighbourhood view (profile card only — charts still city-wide for context)
    if hood_filter:
        _name_col = _col(df_scored, "neighborhood_name", "neighborhood")
        row = df_scored[df_scored[_name_col].str.upper() == hood_filter.upper()]
        if not row.empty:
            r = row.iloc[0]
            grade      = r.get("housing_grade", "—")
            score      = r.get("housing_score")
            rent       = r.get("avg_estimated_rent") or r.get("avg_monthly_rent")
            psqft      = r.get("avg_price_per_sqft")
            sqft       = r.get("avg_living_area_sqft")
            val        = r.get("avg_assessed_value")
            age        = r.get("avg_property_age")
            beds       = r.get("avg_bedrooms")
            baths      = r.get("avg_full_baths")
            n_props    = r.get("total_properties")
            condos     = r.get("condo_count")
            rentals    = r.get("rental_count")
            pct_park   = r.get("pct_has_parking")
            pct_ac     = r.get("pct_has_ac")
            amenity    = r.get("amenity_rate")
            pct_good   = r.get("pct_good_condition")

            grade_bg = {"AFFORDABLE":"#1E8449","AVERAGE":"#b45309",
                        "PREMIUM":"#991b1b","BELOW_AVERAGE":"#475569"}.get(grade, "#475569")

            st.markdown(
                f'<div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);'
                f'border-radius:16px;padding:1.4rem 1.8rem;margin-bottom:1rem;'
                f'display:flex;justify-content:space-between;align-items:center;">'

                f'<div style="flex:1;">'
                f'<div style="font-family:DM Serif Display,serif;font-size:1.8rem;'
                f'color:#e2e8f0;margin-bottom:4px;">🏠 {hood_filter} — Housing</div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">'
                f'<span style="background:{grade_bg};color:#fff;padding:4px 14px;'
                f'border-radius:999px;font-size:11px;font-weight:700;">{grade}</span>'
                f'{"<span style=background:rgba(34,197,94,0.2);color:#22c55e;border:1px solid rgba(34,197,94,0.4);padding:4px 14px;border-radius:999px;font-size:11px;font-weight:700;>" + str(int(n_props)) + " properties</span>" if n_props else ""}'
                f'</div></div>'

                f'<div style="text-align:center;padding-left:2rem;">'
                f'<div style="font-family:DM Serif Display,serif;font-size:3.2rem;'
                f'color:#e2e8f0;line-height:1;">{int(score) if score == score else "—"}</div>'
                f'<div style="color:rgba(255,255,255,0.4);font-size:10px;margin-top:2px;'
                f'text-transform:uppercase;letter-spacing:0.08em;">Affordability Score</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            def _mini(col, lbl, val, sub=""):
                col.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{lbl}</div>'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-sub">{sub}</div></div>',
                    unsafe_allow_html=True,
                )
            _mini(c1, "Avg Monthly Rent",
                  f'${rent:,.0f}/mo' if rent and rent == rent else "—",
                  f'est. monthly rent')
            _mini(c2, "Price / sqft",
                  f'${psqft:.2f}' if psqft == psqft else "—",
                  f'{int(sqft):,} sqft avg' if sqft == sqft else "")
            _mini(c3, "Assessed Value",
                  f'${val:,.0f}' if val == val else "—",
                  f'${psqft:.0f}/sqft' if psqft == psqft else "avg assessed value")
            _mini(c4, "Avg Living Area",
                  f'{int(sqft):,} sqft' if sqft == sqft else "—",
                  f'{beds:.1f} bd · {baths:.1f} ba · {int(age)} yrs old' if (beds == beds and baths == baths and age == age) else "")
            _mini(c5, "Stock Mix",
                  f'{int(condos):,} condos' if condos == condos else "—",
                  f'{int(rentals):,} rentals' if rentals == rentals else "")
            _mini(c6, "Amenities",
                  f'{amenity:.0f}% rate' if amenity == amenity else "—",
                  f'AC {pct_ac:.0f}% · Park {pct_park:.0f}%' if (pct_ac == pct_ac and pct_park == pct_park) else "")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # ── AI Narrative from mart ────────────────────────────────────────
            narrative_txt = r.get("row_description") or r.get("description") or ""
            # Strip leading newlines that sometimes appear in the CSV field
            narrative_txt = narrative_txt.strip()
            if narrative_txt:
                st.markdown(
                    f'<div style="background:rgba(96,165,250,0.07);'
                    f'border:1px solid rgba(96,165,250,0.18);'
                    f'border-left:4px solid #60a5fa;'
                    f'padding:0.75rem 1rem;border-radius:10px;margin-bottom:0.6rem;'
                    f'font-size:0.85rem;line-height:1.65;color:#e2e8f0;">'
                    f'<div style="font-family:DM Serif Display,serif;font-size:0.9rem;'
                    f'color:#93c5fd;margin-bottom:4px;">🏠 Housing Insight — {hood_filter}</div>'
                    f'{narrative_txt}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # No city-wide KPI ribbon — all relevant stats shown in neighbourhood row above

    # When no neighbourhood selected, show city-wide summary ribbon
    if not hood_filter:
        rent_col  = _col(df_scored, "avg_estimated_rent", "avg_monthly_rent")
        psqft_col = _col(df_scored, "avg_price_per_sqft")
        sqft_col  = _col(df_scored, "avg_living_area_sqft")

        total_props  = int(df_scored["total_properties"].sum()) if "total_properties" in df_scored.columns else 0
        median_rent  = df_scored[rent_col].median()  if rent_col  in df_scored.columns else None
        median_psqft = df_scored[psqft_col].median() if psqft_col in df_scored.columns else None
        median_sqft  = df_scored[sqft_col].median()  if sqft_col  in df_scored.columns else None
        affordable_n = int((df_scored["housing_grade"] == "AFFORDABLE").sum()) if "housing_grade" in df_scored.columns else 0
        premium_n    = int((df_scored["housing_grade"] == "PREMIUM").sum())    if "housing_grade" in df_scored.columns else 0

        _kpi_ribbon([
            ("Total Properties",     f'{total_props:,}',                                   "across all neighbourhoods"),
            ("Median Rent",          f'${median_rent:,.0f}/mo'  if median_rent  else "—",  "city-wide median"),
            ("Median $/sqft",        f'${median_psqft:.0f}'     if median_psqft else "—",  "city-wide median"),
            ("Median Living Area",   f'{median_sqft:,.0f} sqft' if median_sqft  else "—",  "avg home size"),
            ("Affordable / Premium", f'{affordable_n} / {premium_n}',                      "grade counts (scored)"),
        ])



    # ══════════════════════════════════════════════════════════════════════════
    # BRANCHED CHARTS: neighbourhood-specific vs city-wide
    # ══════════════════════════════════════════════════════════════════════════

    if hood_filter:
        # ── Pull the single row for this neighbourhood ────────────────────────
        _nc = _col(df_scored, "neighborhood_name", "neighborhood")
        _r  = df_scored[df_scored[_nc].str.upper() == hood_filter.upper()]
        if _r.empty:
            st.info("No detailed data found for this neighbourhood.")
            return
        r = _r.iloc[0]

        # Safe float accessor
        def _f(key, default=0.0):
            v = r.get(key)
            try:
                return float(v) if v not in (None, "", "nan") else default
            except Exception:
                return default

        # ── ROW A: Stock Mix donut  |  Property Age timeline ─────────────────
        col_a1, col_a2 = st.columns(2, gap="medium")

        with col_a1:
            _section("Housing Stock Composition",
                     "Condo · Rental · Other/Owner-occupied breakdown")
            condo_n  = _f("condo_count")
            rental_n = _f("rental_count")
            total_n  = _f("total_properties", 1)
            other_n  = max(total_n - condo_n - rental_n, 0)
            df_stock = pd.DataFrame([
                {"Type": "Condo",       "Count": condo_n,  "Pct": round(condo_n  / total_n * 100, 1)},
                {"Type": "Rental",      "Count": rental_n, "Pct": round(rental_n / total_n * 100, 1)},
                {"Type": "Other/Owner", "Count": other_n,  "Pct": round(other_n  / total_n * 100, 1)},
            ]).query("Count > 0")
            # Only label slices big enough to be readable
            df_stock["Label"] = df_stock["Pct"].apply(
                lambda p: f"{p:.1f}%" if p >= 5 else ""
            )
            donut = alt.Chart(df_stock).mark_arc(
                innerRadius=55, outerRadius=100, stroke="#1a1a2e", strokeWidth=2,
            ).encode(
                theta=alt.Theta("Count:Q", stack=True),
                color=alt.Color("Type:N",
                                scale=alt.Scale(domain=["Condo","Rental","Other/Owner"],
                                                range=["#60a5fa","#34d399","#fbbf24"]),
                                legend=alt.Legend(title=None, orient="bottom",
                                                  direction="horizontal", labelFontSize=12)),
                tooltip=["Type:N",
                         alt.Tooltip("Count:Q", format=","),
                         alt.Tooltip("Pct:Q",   format=".1f", title="%")],
            )
            labels = alt.Chart(df_stock).mark_text(
                radius=120, fontSize=12, fontWeight="bold", color="#e2e8f0",
                align="center", baseline="middle",
            ).encode(
                theta=alt.Theta("Count:Q", stack=True),
                text=alt.Text("Label:N"),
            )
            st.altair_chart(alt.layer(donut, labels).properties(height=340),
                            use_container_width=True)
            st.markdown(
                f'<div style="display:flex;gap:20px;font-size:12px;'
                f'color:rgba(255,255,255,0.55);margin-top:4px;">'
                f'<span>🔵 {int(condo_n):,} condos</span>'
                f'<span>🟢 {int(rental_n):,} rentals</span>'
                f'<span>🟡 {int(other_n):,} other</span>'
                f'<span style="margin-left:auto;color:#e2e8f0;font-weight:600;">'
                f'{int(total_n):,} total</span></div>',
                unsafe_allow_html=True,
            )
            _end_section()

        with col_a2:
            _section("Property Age Span",
                     "Oldest · Average · Newest year built in this neighbourhood")
            oldest  = int(_f("oldest_year_built", 1800))
            newest  = int(_f("newest_year_built", 2024))
            avg_age = _f("avg_property_age", 0)
            avg_yr  = 2025 - int(avg_age) if avg_age else (oldest + newest) // 2
            # Clamp bad data
            if newest > 2025: newest = 2025
            if oldest < 1600: oldest = 1600

            df_pts = pd.DataFrame([
                {"Label": "Oldest Built",  "Year": oldest, "Color": "#b45309", "Size": 130},
                {"Label": "Avg Year Built","Year": avg_yr,  "Color": "#f59e0b", "Size": 200},
                {"Label": "Newest Built",  "Year": newest, "Color": "#67e8f9", "Size": 130},
            ])
            domain_pad = [oldest - 10, newest + 10]
            span = alt.Chart(pd.DataFrame([{"x1": oldest, "x2": newest}])).mark_rule(
                strokeWidth=6, color="#334155",
            ).encode(
                x=alt.X("x1:Q", scale=alt.Scale(domain=domain_pad)),
                x2="x2:Q",
                y=alt.value(130),
            )
            pts = alt.Chart(df_pts).mark_point(filled=True, opacity=1).encode(
                x=alt.X("Year:Q", scale=alt.Scale(domain=domain_pad),
                        axis=alt.Axis(title="Year Built", format="d", labelAngle=0)),
                y=alt.value(130),
                color=alt.Color("Color:N", scale=None, legend=None),
                size=alt.Size("Size:Q", scale=None, legend=None),
                tooltip=["Label:N", alt.Tooltip("Year:Q", format="d")],
            )
            txt_above = alt.Chart(df_pts).mark_text(
                dy=-20, fontSize=12, fontWeight="bold", color="#e2e8f0",
            ).encode(
                x=alt.X("Year:Q", scale=alt.Scale(domain=domain_pad)),
                y=alt.value(130),
                text=alt.Text("Year:Q", format="d"),
            )
            txt_below = alt.Chart(df_pts).mark_text(
                dy=24, fontSize=10, color="rgba(255,255,255,0.45)",
            ).encode(
                x=alt.X("Year:Q", scale=alt.Scale(domain=domain_pad)),
                y=alt.value(130),
                text="Label:N",
            )
            st.altair_chart(
                alt.layer(span, pts, txt_above, txt_below).properties(height=200),
                use_container_width=True,
            )
            st.caption(
                f"**{newest - oldest} year span** · avg age: **{int(avg_age)} yrs** "
                f"· avg built ~**{avg_yr}**"
            )
            _end_section()

        # ── ROW B: Housing Score & Ranking  |  Amenity rates ─────────────────
        col_b1, col_b2 = st.columns(2, gap="medium")

        with col_b1:
            _section("Housing Score & Ranking",
                     f"{hood_filter} · affordability score · city rank")

            _nc2    = _col(df_scored, "neighborhood_name", "neighborhood")
            _rent_c = _col(df_scored, "avg_estimated_rent", "avg_monthly_rent")

            # Compute rank
            df_ranked = df_scored[df_scored["housing_score"].notna()].copy()
            df_ranked = df_ranked.sort_values("housing_score", ascending=False).reset_index(drop=True)
            rank_rows = df_ranked[df_ranked[_nc2].str.upper() == hood_filter.upper()]
            h_rank    = int(rank_rows.index[0]) + 1 if not rank_rows.empty else "—"
            h_total   = len(df_ranked)
            h_score   = _f("housing_score")
            h_grade   = r.get("housing_grade", "—")

            GRADE_COLORS_H = {
                "AFFORDABLE":    ("#1E8449", "#fff"),
                "AVERAGE":       ("#b45309", "#fff"),
                "PREMIUM":       ("#991b1b", "#fff"),
                "BELOW_AVERAGE": ("#475569", "#fff"),
            }
            g_bg, g_fg = GRADE_COLORS_H.get(h_grade, ("#475569", "#fff"))

            # Score + rank banner
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:20px;'
                f'padding:14px 0 10px;border-bottom:1px solid rgba(255,255,255,0.08);">'
                f'<div style="text-align:center;min-width:80px;">'
                f'<div style="font-family:DM Serif Display,serif;font-size:3rem;'
                f'color:#e2e8f0;line-height:1;">{int(h_score) if h_score else "—"}</div>'
                f'<div style="color:rgba(255,255,255,0.3);font-size:9px;'
                f'text-transform:uppercase;letter-spacing:0.08em;">/ 100</div>'
                f'</div>'
                f'<div style="flex:1;">'
                f'<span style="background:{g_bg};color:{g_fg};padding:4px 14px;'
                f'border-radius:999px;font-size:11px;font-weight:700;">{h_grade}</span>'
                f'<div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:6px;">'
                f'Ranked <b style="color:#e2e8f0;">#{h_rank}</b> of '
                f'<b style="color:#e2e8f0;">{h_total}</b> neighbourhoods</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            # Key stat rows — Transit style
            _rent_val = _f(_rent_c)
            _psqft    = _f("avg_price_per_sqft")
            _sqft     = _f("avg_living_area_sqft")
            _val      = _f("avg_assessed_value")
            _beds     = _f("avg_bedrooms")
            _baths    = _f("avg_full_baths")
            _age      = _f("avg_property_age")
            _density  = _f("properties_per_sqmile")
            _mn       = _f("min_assessed_value")
            _mx       = _f("max_assessed_value")

            stat_rows = [
                ("💰", "Avg Monthly Rent",    f'${_rent_val:,.0f}/mo'              if _rent_val         else "—"),
                ("📐", "Price per sqft",      f'${_psqft:.2f}'                     if _psqft            else "—"),
                ("🏠", "Avg Living Area",     f'{int(_sqft):,} sqft'               if _sqft             else "—"),
                ("🏦", "Avg Assessed Value",  f'${_val:,.0f}'                      if _val              else "—"),
                ("🛏️", "Bedrooms / Baths",   f'{_beds:.1f} bd · {_baths:.1f} ba'  if (_beds and _baths) else "—"),
                ("📅", "Avg Property Age",   f'{int(_age)} years old'              if _age              else "—"),
                ("📊", "Value Range",         f'${_mn:,.0f} – ${_mx:,.0f}'         if (_mn and _mx)     else "—"),
                ("🗺️", "Density",            f'{_density:,.0f} props/sqmi'         if _density          else "—"),
            ]
            html_stats = '<div style="margin-top:4px;">'
            for icon, label, val in stat_rows:
                html_stats += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:7px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<span style="color:rgba(255,255,255,0.4);font-size:12px;">{icon} {label}</span>'
                    f'<span style="color:#e2e8f0;font-size:12px;font-weight:600;">{val}</span>'
                    f'</div>'
                )
            html_stats += '</div>'
            st.markdown(html_stats, unsafe_allow_html=True)
            _end_section()

        with col_b2:
            _section("Amenity Rates",
                     f"% of {hood_filter} properties with each amenity")
            pct_ac     = _f("pct_has_ac")
            pct_park   = _f("pct_has_parking")
            fp_count   = _f("has_fireplace_count")
            tot_am     = _f("total_properties", 1)
            pct_fire   = round(fp_count / tot_am * 100, 1) if tot_am > 0 else 0.0
            amenity_rt = _f("amenity_rate")

            df_am = pd.DataFrame([
                {"Amenity": "Air Conditioning", "Pct": pct_ac,   "Color": "#60a5fa"},
                {"Amenity": "Parking",          "Pct": pct_park, "Color": "#34d399"},
                {"Amenity": "Fireplace",        "Pct": pct_fire, "Color": "#f97316"},
            ])
            if df_am["Pct"].sum() > 0:
                bars_am = alt.Chart(df_am).mark_bar(
                    cornerRadiusTopRight=5, cornerRadiusBottomRight=5,
                ).encode(
                    y=alt.Y("Amenity:N",
                            sort=["Air Conditioning","Parking","Fireplace"],
                            axis=alt.Axis(title=None, labelFontSize=13, labelFontWeight="bold")),
                    x=alt.X("Pct:Q", scale=alt.Scale(domain=[0, 100]),
                            axis=alt.Axis(title="% of Properties")),
                    color=alt.Color("Color:N", scale=None, legend=None),
                    tooltip=["Amenity:N", alt.Tooltip("Pct:Q", format=".1f", title="%")],
                )
                txt_am = alt.Chart(df_am).mark_text(
                    align="left", dx=4, fontSize=13, fontWeight="bold", color="#e2e8f0",
                ).encode(
                    y=alt.Y("Amenity:N", sort=["Air Conditioning","Parking","Fireplace"]),
                    x=alt.X("Pct:Q"),
                    text=alt.Text("Pct:Q", format=".1f"),
                )
                st.altair_chart(alt.layer(bars_am, txt_am).properties(height=200),
                                use_container_width=True)
                st.caption(f"Overall amenity rate: **{amenity_rt:.1f}%**")
            else:
                st.info("Amenity data not available for this neighbourhood.")
            _end_section()

        # ── ROW C: Property stock breakdown  |  Neighbourhood profile card ───
        col_c1, col_c2 = st.columns(2, gap="medium")

        with col_c1:
            _section("Property Stock Breakdown",
                     f"Full housing inventory breakdown for {hood_filter}")

            total_n  = _f("total_properties", 1)
            condo_n  = _f("condo_count")
            rental_n = _f("rental_count")
            other_n  = max(total_n - condo_n - rental_n, 0)

            # Stock type cards — Bluebikes station-list style
            STOCK_TYPES = [
                ("🏢", "Condominiums",       condo_n,  "#60a5fa",
                 f'{condo_n / total_n * 100:.1f}% of total stock'),
                ("🏘️", "Rental Properties", rental_n, "#34d399",
                 f'{rental_n / total_n * 100:.1f}% of total stock'),
                ("🏡", "Other / Owner-Occ.", other_n,  "#fbbf24",
                 f'{other_n / total_n * 100:.1f}% of total stock'),
            ]
            html_stock = '<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:12px;">'
            for icon, label, count, color, sub in STOCK_TYPES:
                bar_pct = max(4, int(count / total_n * 100)) if total_n > 0 else 0
                html_stock += (
                    f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
                    f'border-radius:10px;padding:10px 12px;position:relative;overflow:hidden;">'
                    f'<div style="position:absolute;bottom:0;left:0;height:3px;'
                    f'width:{bar_pct}%;background:{color};opacity:0.7;"></div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="color:#e2e8f0;font-size:12px;font-weight:600;">{icon} {label}</span>'
                    f'<span style="color:{color};font-size:16px;font-weight:700;">{int(count):,}</span>'
                    f'</div>'
                    f'<div style="color:rgba(255,255,255,0.3);font-size:10px;margin-top:3px;">{sub}</div>'
                    f'</div>'
                )
            html_stock += '</div>'
            # Total row
            html_stock += (
                f'<div style="display:flex;justify-content:space-between;padding:8px 12px;'
                f'background:rgba(255,255,255,0.06);border-radius:8px;margin-bottom:12px;">'
                f'<span style="color:rgba(255,255,255,0.5);font-size:12px;">Total Properties</span>'
                f'<span style="color:#e2e8f0;font-size:14px;font-weight:700;">{int(total_n):,}</span>'
                f'</div>'
            )
            st.markdown(html_stock, unsafe_allow_html=True)

            # Condition breakdown bars
            cond_data = [
                ("✨", "Excellent", _f("excellent_count"), "#22c55e"),
                ("👍", "Very Good", _f("very_good_count"), "#86efac"),
                ("✅", "Good",      _f("good_count"),      "#fbbf24"),
                ("➖", "Average",   _f("average_count"),   "#f97316"),
                ("⚠️", "Fair",      _f("fair_count"),      "#ef4444"),
            ]
            cond_total = sum(c for _, _, c, _ in cond_data)
            if cond_total > 0:
                st.markdown(
                    '<div style="color:rgba(255,255,255,0.35);font-size:10px;'
                    'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">'
                    'Condition Quality</div>',
                    unsafe_allow_html=True,
                )
                html_cond = '<div style="display:flex;flex-direction:column;gap:4px;">'
                for icon, label, count, color in cond_data:
                    if count == 0:
                        continue
                    pct = count / cond_total * 100
                    bar = max(2, int(pct))
                    html_cond += (
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<span style="color:rgba(255,255,255,0.4);font-size:11px;min-width:70px;">'
                        f'{icon} {label}</span>'
                        f'<div style="flex:1;background:rgba(255,255,255,0.06);'
                        f'border-radius:3px;height:14px;position:relative;overflow:hidden;">'
                        f'<div style="position:absolute;left:0;top:0;height:100%;'
                        f'width:{bar}%;background:{color};opacity:0.75;"></div>'
                        f'<div style="position:absolute;left:5px;top:50%;'
                        f'transform:translateY(-50%);color:#fff;font-size:9px;font-weight:600;">'
                        f'{int(count):,} ({pct:.1f}%)</div>'
                        f'</div></div>'
                    )
                html_cond += '</div>'
                st.markdown(html_cond, unsafe_allow_html=True)
            _end_section()

        with col_c2:
            _section("Neighbourhood Profile",
                     f"{hood_filter} · location · area · character")

            dist_group  = str(r.get("distribution_group", "—")).replace("_", " ").title()
            city_name   = str(r.get("city", "—")).title()
            sqmiles_val = _f("sqmiles")
            oldest_yr   = int(_f("oldest_year_built", 0))
            newest_yr   = int(_f("newest_year_built", 0))
            avg_age_val = _f("avg_property_age")
            density_val = _f("properties_per_sqmile")
            pct_good_v  = _f("pct_good_condition")
            avg_cond_v  = _f("avg_condition_score")

            GROUP_COLORS_P = {
                "Boston":         ("#60a5fa", "#1e3a5f"),
                "Cambridge":      ("#a78bfa", "#2d1b69"),
                "Greater Boston": ("#34d399", "#064e3b"),
            }
            gc_fg, gc_bg = GROUP_COLORS_P.get(dist_group, ("#e2e8f0", "#334155"))

            # Location pills
            st.markdown(
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">'
                f'<span style="background:{gc_bg};color:{gc_fg};border:1.5px solid {gc_fg}44;'
                f'padding:4px 14px;border-radius:999px;font-size:11px;font-weight:700;">'
                f'📍 {city_name}, Massachusetts</span>'
                f'<span style="background:rgba(255,255,255,0.06);color:rgba(255,255,255,0.6);'
                f'padding:4px 14px;border-radius:999px;font-size:11px;">'
                f'{dist_group}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Profile stat rows — exact Transit/Bluebikes style
            cond_label_v = {1:"Poor",2:"Average",3:"Good",4:"Very Good",5:"Excellent"}.get(
                round(avg_cond_v) if avg_cond_v else 0, "—"
            )
            profile_rows = [
                ("📐", "Area",              f'{sqmiles_val:.2f} sq miles'            if sqmiles_val  else "—"),
                ("🏘️", "Property Density",  f'{density_val:,.0f} properties / sq mi' if density_val  else "—"),
                ("🗓️", "Oldest Property",  f'{oldest_yr}'                            if oldest_yr    else "—"),
                ("🆕", "Newest Property",   f'{newest_yr}'                            if newest_yr    else "—"),
                ("⏳", "Avg Property Age",  f'{int(avg_age_val)} years'               if avg_age_val  else "—"),
                ("🏢", "Condo Share",       f'{_f("condo_count")/total_n*100:.1f}%'  if total_n > 1  else "—"),
                ("🏘️", "Rental Share",      f'{_f("rental_count")/total_n*100:.1f}%' if total_n > 1  else "—"),
                ("⭐", "Avg Condition",     f'{cond_label_v} ({avg_cond_v:.2f}/5)'   if avg_cond_v   else "—"),
                ("✅", "% Good Condition",  f'{pct_good_v:.1f}% of stock'            if pct_good_v   else "—"),
            ]
            html_prof = '<div style="margin-bottom:12px;">'
            for icon, label, val in profile_rows:
                html_prof += (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:7px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<div style="display:flex;align-items:center;gap:6px;">'
                    f'<span style="font-size:13px;">{icon}</span>'
                    f'<span style="color:rgba(255,255,255,0.45);font-size:12px;">{label}</span>'
                    f'</div>'
                    f'<span style="color:#e2e8f0;font-size:12px;font-weight:600;">{val}</span>'
                    f'</div>'
                )
            html_prof += '</div>'
            st.markdown(html_prof, unsafe_allow_html=True)

            # Amenity pills — compact Bluebikes-style
            pct_ac2    = _f("pct_has_ac")
            pct_park2  = _f("pct_has_parking")
            fp_count2  = _f("has_fireplace_count")
            total_n2   = _f("total_properties", 1)
            pct_fire2  = round(fp_count2 / total_n2 * 100, 1) if total_n2 > 0 else 0.0
            amenity_rt2 = _f("amenity_rate")

            if amenity_rt2 > 0:
                st.markdown(
                    '<div style="color:rgba(255,255,255,0.35);font-size:10px;'
                    'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">'
                    'Amenity Rates</div>',
                    unsafe_allow_html=True,
                )
                amenity_items = [
                    ("❄️ AC",       pct_ac2,   "#60a5fa"),
                    ("🚗 Parking",  pct_park2, "#34d399"),
                    ("🔥 Fireplace",pct_fire2, "#f97316"),
                    ("⭐ Overall",   amenity_rt2,"#e2e8f0"),
                ]
                pill_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
                for label, pct, color in amenity_items:
                    pill_html += (
                        f'<div style="background:{color}18;border:1px solid {color}44;'
                        f'border-radius:8px;padding:6px 12px;text-align:center;">'
                        f'<div style="color:{color};font-size:13px;font-weight:700;">{pct:.1f}%</div>'
                        f'<div style="color:rgba(255,255,255,0.4);font-size:9px;'
                        f'text-transform:uppercase;letter-spacing:0.05em;">{label}</div>'
                        f'</div>'
                    )
                pill_html += '</div>'
                st.markdown(pill_html, unsafe_allow_html=True)
            _end_section()


    else:
        # ══════════════════════════════════════════════════════════════════════
        # CITY-WIDE charts — shown when no neighbourhood is selected (ALL)
        # ══════════════════════════════════════════════════════════════════════

        col_left, col_right = st.columns([1.4, 1], gap="medium")

        with col_left:
            _section("Affordability Rankings",
                     "Higher score = more affordable relative to Boston market · top 30")
            st.altair_chart(_chart_affordability_bar(df_scored), use_container_width=True)
            _end_section()

        with col_right:
            _section("Grade Distribution", "All scored neighbourhoods")
            st.altair_chart(_chart_grade_donut(df_scored), use_container_width=True)
            _end_section()

            _section("Rent vs Affordability Score", "Each dot = one neighbourhood")
            st.altair_chart(_chart_rent_vs_score(df_scored), use_container_width=True)
            _end_section()

        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            df_bubble = df_scored[df_scored["avg_bedrooms"].notna()].copy()
            if not df_bubble.empty:
                _section("Space & Size Landscape",
                         "Bubble size = # properties · x = avg sqft · y = avg bedrooms")
                st.altair_chart(_chart_living_area_bubble(df_bubble), use_container_width=True)
                _end_section()
            else:
                _section("Space & Size Landscape", "Bedroom data not available")
                st.info("Bedroom data unavailable.")
                _end_section()

        with col_b:
            _section("Property Age Profile",
                     "Top 30 by avg age · Blue = newer · Amber = historic")
            st.altair_chart(_chart_property_age(df_scored), use_container_width=True)
            _end_section()