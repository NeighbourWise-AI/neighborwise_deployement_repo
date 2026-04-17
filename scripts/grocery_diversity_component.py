"""
grocery_diversity_component.py  — NeighbourWise AI
====================================================
Uses the updated /domain/grocery endpoint which now returns:
  - scores[]        : from GA_GROCERY_HOTSPOT_CLUSTERS (ACCESS_TIER, real store counts)
  - map{}           : GeoJSON FeatureCollection with polygon geometry + fill_color
  - neighbors[]     : adjacent neighborhoods (only when hood_filter set)
  - summary{}       : tier counts, food_desert_count

Layout (single scrolling page, no sub-tabs)
-------------------------------------------
1. Grocery KPI cards
2. Essential count bars  +  Access tier donut
3. GeoJSON polygon map   +  Neighbourhood snapshot / neighbors panel
4. Scorecard table
5. Deep-dive card        (only when sidebar hood selected)
"""

import json
import pandas as pd
import streamlit as st
import altair as alt

STORE_TYPES = [
    ("supermarkets",            "Supermarket",    "#378ADD"),
    ("specialty_food_count",    "Specialty food", "#BA7517"),
    ("farmers_markets",         "Farmers market", "#3B6D11"),
    ("pharmacies",              "Pharmacy",       "#7F77DD"),
    ("convenience_store_count", "Convenience",    "#888780"),
]

TIER_HEX = {
    "HIGH_ACCESS":  "#1E8449",
    "GOOD_ACCESS":  "#2E86C1",
    "FAIR_ACCESS":  "#D4AC0D",
    "LOW_ACCESS":   "#C0392B",
    "FOOD_DESERT":  "#C0392B",
    "DESERT":       "#C0392B",
}

TIER_FILL = {
    "HIGH_ACCESS":  [30,  132, 73,  180],
    "GOOD_ACCESS":  [46,  134, 193, 160],
    "FAIR_ACCESS":  [212, 172, 13,  170],
    "LOW_ACCESS":   [192, 57,  43,  180],
    "FOOD_DESERT":  [192, 57,  43,  180],
    "DESERT":       [192, 57,  43,  180],
}

TIER_ORDER = ["HIGH_ACCESS","GOOD_ACCESS","FAIR_ACCESS","LOW_ACCESS","FOOD_DESERT","DESERT"]


def _build_df(scores: list) -> pd.DataFrame:
    """Build clean DataFrame from scores list. Keeps all rows including 0-store."""
    rows = []
    for r in scores:
        rows.append({
            "neighborhood":          str(r.get("neighborhood", "")),
            "city":                  str(r.get("city", "")),
            "access_tier":           str(r.get("access_tier", "GOOD_ACCESS")).upper(),
            "grocery_score":         float(r.get("grocery_score") or 0),
            "total_stores":          int(r.get("total_stores") or 0),
            "essential_stores":      int(r.get("essential_stores") or 0),
            "pct_essential":         float(r.get("pct_essential") or 0),
            "supermarkets":          int(r.get("supermarkets") or 0),
            "convenience_store_count": int(r.get("convenience_store_count") or 0),
            "specialty_food_count":  int(r.get("specialty_food_count") or 0),
            "pharmacies":            int(r.get("pharmacies") or 0),
            "farmers_markets":       int(r.get("farmers_markets") or 0),
            "n_clusters":            int(r.get("n_clusters") or 0),
            "clustered_share":       float(r.get("clustered_store_share_pct") or 0),
            "isolated_pct":          float(r.get("isolated_store_pct") or 0),
            "stores_per_sqmile":     float(r.get("stores_per_sqmile") or 0),
            "pct_convenience":       float(r.get("pct_convenience") or 0),
            "lat":                   float(r.get("lat") or 0),
            "lon":                   float(r.get("lon") or 0),
            "description":           str(r.get("description") or ""),
            "data_year":             str(r.get("data_year") or "2021"),
            "reliability":           str(r.get("reliability") or ""),
        })
    return pd.DataFrame(rows)


# ── 1. KPI cards ──────────────────────────────────────────────────────────────
def _kpi_cards(df: pd.DataFrame, summary: dict):
    low_cnt = int((df["access_tier"].isin(
        ["LOW_ACCESS","FOOD_DESERT","DESERT"]
    )).sum())
    df_s = df[df["total_stores"] > 0]
    avg_essential = round(df_s["pct_essential"].mean(), 1) if not df_s.empty else 0

    items = [
        ("Neighborhoods",    df["neighborhood"].nunique(),  "In current filter"),
        ("Total Stores",     f'{int(df["total_stores"].sum()):,}', "Unique, deduplicated"),
        ("Low Access Zones", str(low_cnt),                  "Low access / food desert"),
        ("Avg Essential %",  f"{avg_essential}%",           "Supermarkets + produce"),
    ]
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )


# ── 2. Essential count bars + tier donut ──────────────────────────────────────
def _section_overview(df: pd.DataFrame):
    c1, c2 = st.columns([1.2, 1], gap="medium")

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Essential Store Count by Neighborhood</div>',
            unsafe_allow_html=True,
        )
        top12 = (
            df[df["total_stores"] > 0]
            .sort_values("essential_stores", ascending=False)
            .head(12)
        )
        bar = alt.Chart(top12).mark_bar(
            cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#52b788",
        ).encode(
            x=alt.X("essential_stores:Q", title="Essential Stores"),
            y=alt.Y("neighborhood:N", sort="-x",
                    axis=alt.Axis(title=None, labelFontSize=10, labelLimit=160)),
            tooltip=[
                "neighborhood:N",
                alt.Tooltip("essential_stores:Q", title="Essential Stores"),
                alt.Tooltip("total_stores:Q",     title="Total Stores"),
                "access_tier:N",
            ],
        )
        st.altair_chart(bar.properties(height=380), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Access Tier Distribution</div>'
            '<div class="section-subtitle">All neighbourhoods incl. low access</div>',
            unsafe_allow_html=True,
        )
        td = df.groupby("access_tier").size().reset_index(name="count")
        total_n = td["count"].sum()
        td["pct"]       = (td["count"] / total_n * 100).round(1)
        td["pct_str"]   = td["pct"].apply(lambda v: f"{v:.0f}%")
        td["pct_label"] = td.apply(
            lambda r: f'{r["pct"]:.0f}%' if r["pct"] >= 4 else "", axis=1
        )

        donut = alt.Chart(td).mark_arc(
            innerRadius=55, outerRadius=100, stroke="#1a1a2e", strokeWidth=2,
        ).encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color(
                "access_tier:N",
                scale=alt.Scale(
                    domain=list(TIER_HEX.keys()),
                    range=list(TIER_HEX.values()),
                ),
                legend=None,
            ),
            order=alt.Order("count:Q", sort="descending"),
            tooltip=[
                alt.Tooltip("access_tier:N", title="Tier"),
                alt.Tooltip("count:Q",        title="Neighborhoods"),
                alt.Tooltip("pct_str:N",      title="Share"),
            ],
        )
        pct_labels = alt.Chart(td).mark_text(
            radius=78, fontSize=13, fontWeight="bold", color="#ffffff",
        ).encode(
            theta=alt.Theta("count:Q", stack=True),
            order=alt.Order("count:Q", sort="descending"),
            text=alt.Text("pct_label:N"),
        )

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.altair_chart(
                alt.layer(donut, pct_labels)
                   .properties(height=280, width=280)
                   .configure_view(strokeWidth=0),
                use_container_width=False,
            )

        # Custom legend — only tiers present in data
        present = td.sort_values("count", ascending=False)["access_tier"].tolist()
        leg_html = '<div style="display:flex;flex-direction:column;gap:5px;margin-top:8px;">'
        for tier in present:
            row   = td[td["access_tier"] == tier].iloc[0]
            color = TIER_HEX.get(tier, "#64748b")
            label = tier.replace("_", " ").title()
            leg_html += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:3px 6px;border-radius:6px;background:rgba(255,255,255,0.04);">'
                f'<span style="display:flex;align-items:center;gap:7px;">'
                f'<span style="width:10px;height:10px;border-radius:50%;'
                f'background:{color};flex-shrink:0;"></span>'
                f'<span style="font-size:11px;color:#e2e8f0;">{label}</span></span>'
                f'<span style="font-size:11px;color:#94a3b8;">'
                f'{int(row["count"])} &nbsp;<b style="color:#e2e8f0;">{row["pct"]:.1f}%</b>'
                f'</span></div>'
            )
        leg_html += '</div>'
        st.markdown(leg_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── 3. GeoJSON polygon map + snapshot panel ───────────────────────────────────
def _section_map(domain_data: dict, df: pd.DataFrame, hood_filter):
    try:
        import pydeck as pdk
    except ImportError:
        st.warning("pydeck not installed — map unavailable.")
        return

    map_geojson = domain_data.get("map", {})
    neighbors   = domain_data.get("neighbors", [])
    features    = map_geojson.get("features", [])

    col_map, col_detail = st.columns([1.6, 1], gap="medium")

    with col_map:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Boston Food Access Map</div>'
            + (
                f'<div class="section-subtitle">Viewing: <b>{hood_filter}</b> '
                f'— highlighted with white border</div>'
                if hood_filter
                else '<div class="section-subtitle">Polygons shaded by access tier · '
                     'green = high access · amber = fair · red = low access</div>'
            ),
            unsafe_allow_html=True,
        )

        if not features:
            st.info("Map geometry not available.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Mark selected neighbourhood
            if hood_filter:
                for f in features:
                    f["properties"]["is_selected"] = (
                        f["properties"]["neighborhood"].upper() == hood_filter.upper()
                    )

            # Set zoom/center
            if hood_filter:
                sel = next(
                    (f for f in features
                     if f["properties"]["neighborhood"].upper() == hood_filter.upper()),
                    None,
                )
                if sel and sel["properties"].get("latitude"):
                    ctr_lat = sel["properties"]["latitude"]
                    ctr_lon = sel["properties"]["longitude"]
                    zoom    = 11.8   # shows selected + all surrounding neighbours
                else:
                    lats = [f["properties"].get("latitude", 0) for f in features if f["properties"].get("latitude")]
                    lons = [f["properties"].get("longitude", 0) for f in features if f["properties"].get("longitude")]
                    ctr_lat = sum(lats)/len(lats) if lats else 42.36
                    ctr_lon = sum(lons)/len(lons) if lons else -71.06
                    zoom    = 10.8
            else:
                lats = [f["properties"].get("latitude", 0) for f in features if f["properties"].get("latitude")]
                lons = [f["properties"].get("longitude", 0) for f in features if f["properties"].get("longitude")]
                ctr_lat = sum(lats)/len(lats) if lats else 42.36
                ctr_lon = sum(lons)/len(lons) if lons else -71.06
                zoom    = 10.8

            # ── Layer 1: All neighbourhood polygons coloured by tier ────────
            layers = [
                pdk.Layer(
                    "GeoJsonLayer",
                    data={"type": "FeatureCollection", "features": features},
                    filled=True, stroked=True, pickable=True, auto_highlight=True,
                    get_fill_color="properties.fill_color",
                    get_line_color=[255, 255, 255, 80],
                    line_width_min_pixels=1,
                )
            ]

            # ── Layer 2: White highlight border for selected neighbourhood ────
            if hood_filter:
                sel_feats = [f for f in features if f["properties"].get("is_selected")]
                if sel_feats:
                    layers.append(pdk.Layer(
                        "GeoJsonLayer",
                        data={"type": "FeatureCollection", "features": sel_feats},
                        filled=False, stroked=True,
                        get_line_color=[255, 255, 255, 255],
                        line_width_min_pixels=3,
                        pickable=False,
                    ))

            # ── Layer 3: Store scatter dots ──────────────────────────────────
            # Split into 4 sub-layers by (essential x selected) so pydeck
            # can use hard-coded RGBA without needing list-column resolution
            store_points = domain_data.get("store_points", [])
            if store_points:
                import pandas as _pd

                sdf = _pd.DataFrame(store_points)
                # Force essential to proper bool so ~ works correctly
                sdf["essential"] = sdf["essential"].apply(
                    lambda v: bool(v) if not isinstance(v, bool) else v
                )
                # Normalise the neighborhood name column
                sdf["_nbhd"] = sdf["neighborhood"].str.upper().str.strip()
                _hf_upper = hood_filter.upper().strip() if hood_filter else ""

                def _add_scatter(subset, color, radius, line_color=None):
                    if subset.empty:
                        return
                    layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=subset[["lon", "lat", "name", "store_type", "neighborhood"]],
                        get_position=["lon", "lat"],
                        get_fill_color=color,
                        get_line_color=line_color or color,
                        get_radius=radius,
                        radius_min_pixels=2,
                        radius_max_pixels=8,
                        pickable=True,
                        stroked=False,
                    ))

                if hood_filter:
                    # Selected neighbourhood — bright dots on top
                    sel_ess  = sdf[(sdf["_nbhd"] == _hf_upper) &  sdf["essential"]]
                    sel_non  = sdf[(sdf["_nbhd"] == _hf_upper) & ~sdf["essential"]]
                    # All other neighbourhoods — dimmed
                    oth_ess  = sdf[(sdf["_nbhd"] != _hf_upper) &  sdf["essential"]]
                    oth_non  = sdf[(sdf["_nbhd"] != _hf_upper) & ~sdf["essential"]]

                    _add_scatter(oth_non,  [180, 180, 180,  45], 30)
                    _add_scatter(oth_ess,  [134, 197, 164,  60], 40)
                    _add_scatter(sel_non,  [220, 220, 220, 200], 35)
                    _add_scatter(sel_ess,  [134, 197, 164, 240], 55)
                else:
                    # All neighbourhoods — uniform medium opacity
                    non_ess = sdf[~sdf["essential"]]
                    ess     = sdf[ sdf["essential"]]
                    _add_scatter(non_ess, [180, 180, 180, 80],  30)
                    _add_scatter(ess,     [134, 197, 164, 100], 40)

            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=pdk.ViewState(
                        latitude=ctr_lat, longitude=ctr_lon, zoom=zoom, pitch=0,
                    ),
                    tooltip={
                        "html": "<b>{neighborhood}</b><br/>"
                                "<span style='opacity:0.7'>{name}</span><br/>"
                                "Tier: <b>{access_tier_label}</b> &nbsp; "
                                "🏪 Total: <b>{total_stores}</b><br/>"
                                "🥦 Essential: <b>{essential_stores}</b> &nbsp; "
                                "🏬 <b>{supermarkets}</b> supermarkets",
                        "style": {
                            "backgroundColor": "#1e293b",
                            "color": "#e2e8f0",
                            "fontSize": "12px",
                            "borderRadius": "8px",
                            "padding": "10px",
                        },
                    },
                    map_style="mapbox://styles/mapbox/dark-v10",
                ),
                use_container_width=True, height=520,
            )

            l1, l2, l3, l4 = st.columns(4)
            l1.markdown('<span style="color:#1E8449;">■</span> **High Access**',  unsafe_allow_html=True)
            l2.markdown('<span style="color:#2E86C1;">■</span> **Good Access**',  unsafe_allow_html=True)
            l3.markdown('<span style="color:#D4AC0D;">■</span> **Fair Access**',  unsafe_allow_html=True)
            l4.markdown('<span style="color:#C0392B;">■</span> **Low Access**',   unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Right panel ────────────────────────────────────────────────────────────
    with col_detail:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        if hood_filter and hood_filter.upper() in df["neighborhood"].str.upper().values:
            r = df[df["neighborhood"].str.upper() == hood_filter.upper()].iloc[0]
            tier_color = TIER_HEX.get(r["access_tier"], "#64748b")
            tier_label = r["access_tier"].replace("_", " ").title()

            st.markdown(
                f'<div class="section-title">🛒 {r["neighborhood"]} — Snapshot</div>'
                f'<div class="section-subtitle">{r["city"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span style="background:{tier_color}22;color:{tier_color};'
                f'border:1px solid {tier_color}44;padding:3px 10px;border-radius:999px;'
                f'font-size:11px;font-weight:600;">{tier_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            for label, val in [
                ("Total Stores",       str(int(r["total_stores"]))),
                ("Essential Stores",   f'{int(r["essential_stores"])} ({r["pct_essential"]:.1f}%)'),
                ("Supermarkets",       str(int(r["supermarkets"]))),
                ("Convenience",        str(int(r["convenience_store_count"]))),
                ("Specialty",          str(int(r["specialty_food_count"]))),
                ("Pharmacies",         str(int(r["pharmacies"]))),
                ("Farmers Markets",    str(int(r["farmers_markets"]))),
                ("DBSCAN Clusters",    str(int(r["n_clusters"]))),
                ("Clustered Share",    f'{r["clustered_share"]:.1f}%'),
                ("Isolated Stores",    f'{r["isolated_pct"]:.1f}%'),
            ]:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
                    f'<span style="color:rgba(255,255,255,0.45);font-size:12px;">{label}</span>'
                    f'<span style="color:#e2e8f0;font-size:12px;font-weight:500;">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # AI Narrative
            if r.get("description"):
                st.markdown(
                    f'<div class="narrative-box" style="margin-top:12px;">'
                    f'<div class="narrative-title">AI Narrative</div>'
                    f'{str(r["description"])}'
                    f'<div style="margin-top:6px;font-size:10px;'
                    f'color:rgba(255,255,255,0.3);">'
                    f'Data year: {r["data_year"]} · Reliability: {r["reliability"]}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # Neighboring areas
            if neighbors:
                st.markdown(
                    '<div style="margin-top:12px;font-size:11px;font-weight:600;'
                    'color:rgba(255,255,255,0.4);text-transform:uppercase;'
                    'letter-spacing:.05em;margin-bottom:6px;">Neighboring Areas</div>',
                    unsafe_allow_html=True,
                )
                df_nb = pd.DataFrame(neighbors)
                st.dataframe(
                    df_nb[["neighborhood","essential_stores","total_stores","access_tier"]]
                      .rename(columns={
                          "neighborhood":   "Neighborhood",
                          "essential_stores":"Essential",
                          "total_stores":   "Total",
                          "access_tier":    "Tier",
                      }),
                    use_container_width=True, hide_index=True,
                )

        else:
            st.markdown(
                '<div class="section-title">Neighborhood Detail</div>'
                '<div class="section-subtitle">Select a neighborhood from the sidebar '
                'to see its snapshot, AI narrative, and neighboring area comparison.</div>',
                unsafe_allow_html=True,
            )
            for label, ascending in [
                ("Top 5 — Highest Essential Count",   False),
                ("Bottom 5 — Lowest Essential Count",  True),
            ]:
                st.markdown(f"**{label}**")
                st.dataframe(
                    df.sort_values("essential_stores", ascending=ascending)
                      .head(5)[["neighborhood","essential_stores","access_tier"]]
                      .rename(columns={
                          "neighborhood":   "Neighborhood",
                          "essential_stores":"Essential",
                          "access_tier":    "Tier",
                      }),
                    use_container_width=True, hide_index=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)


# ── 4. Scorecard table ────────────────────────────────────────────────────────
def _section_scorecard(df: pd.DataFrame):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Neighborhood Food Access Scorecard</div>',
        unsafe_allow_html=True,
    )
    display = {
        "neighborhood":           "Neighborhood",
        "city":                   "City",
        "total_stores":           "Total Stores",
        "essential_stores":       "Essential",
        "pct_essential":          "Essential %",
        "supermarkets":           "Supermarkets",
        "convenience_store_count":"Convenience",
        "specialty_food_count":   "Specialty",
        "n_clusters":             "Clusters",
        "access_tier":            "Access Tier",
        "grocery_score":          "Score",
    }
    present = [c for c in display if c in df.columns]
    st.dataframe(
        df[present]
          .rename(columns={c: display[c] for c in present})
          .sort_values("Essential", ascending=False),
        use_container_width=True, hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ── 5. Deep-dive card ─────────────────────────────────────────────────────────
def _section_deep_dive(df: pd.DataFrame, hood_filter):
    if not hood_filter:
        st.markdown(
            '<div class="narrative-box-blue">'
            '<div class="narrative-title">💡 Tip</div>'
            'Select a specific neighbourhood from the <b>Neighborhood</b> dropdown '
            'in the sidebar to see a detailed store-type breakdown for that area.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    match = df[df["neighborhood"].str.upper() == hood_filter.upper()]
    if match.empty:
        return

    r           = match.iloc[0]
    tier_color  = TIER_HEX.get(r["access_tier"], "#64748b")
    tier_label  = r["access_tier"].replace("_", " ").title()
    total       = int(r["total_stores"])

    type_rows = sorted(
        [(lbl, int(r.get(col, 0) or 0), clr)
         for col, lbl, clr in STORE_TYPES if int(r.get(col, 0) or 0) > 0],
        key=lambda x: -x[1],
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">🛒 {r["neighborhood"]} — Grocery Deep Dive</div>'
        f'<div class="section-subtitle">{r["city"]} · Data year {r["data_year"]}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, sub in [
        (c1, "Access Tier",
              tier_label,
              f'<span style="background:{tier_color}22;color:{tier_color};'
              f'border:1px solid {tier_color}44;padding:2px 8px;border-radius:999px;'
              f'font-size:10px;font-weight:600;">{tier_label}</span>'),
        (c2, "Total Stores",   str(total),                    f'{r["stores_per_sqmile"]:.1f}/sq mi'),
        (c3, "Essential %",    f'{r["pct_essential"]:.1f}%',  f'{r["essential_stores"]} stores'),
        (c4, "DBSCAN Clusters",str(int(r["n_clusters"])),     f'{r["clustered_share"]:.1f}% clustered'),
        (c5, "Convenience %",  f'{r["pct_convenience"]:.1f}%',"of total stores"),
    ]:
        col.markdown(
            f'<div style="background:rgba(255,255,255,0.04);padding:10px 12px;'
            f'border-radius:10px;border:1px solid rgba(255,255,255,0.08);margin-bottom:8px;">'
            f'<div style="font-size:10px;font-weight:600;color:rgba(255,255,255,0.45);'
            f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">{label}</div>'
            f'<div style="font-family:DM Serif Display,serif;font-size:1.1rem;'
            f'color:#e2e8f0;line-height:1.2;">{value}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,0.35);margin-top:3px;">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if total == 0:
        st.markdown(
            '<div class="narrative-box" style="border-left-color:#C0392B;'
            'background:rgba(192,57,43,0.08);">'
            '<div class="narrative-title" style="color:#f87171;">🚨 Food Desert</div>'
            'This neighborhood has no grocery or food retail stores.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="margin-top:4px;font-size:11px;font-weight:600;'
            'color:rgba(255,255,255,0.4);text-transform:uppercase;'
            'letter-spacing:.05em;margin-bottom:6px;">Store type breakdown</div>',
            unsafe_allow_html=True,
        )
        for lbl, cnt, clr in type_rows:
            pct = round(cnt / total * 100, 1) if total else 0
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;'
                f'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<span style="width:10px;height:10px;border-radius:2px;'
                f'background:{clr};flex-shrink:0;"></span>'
                f'<span style="color:#e2e8f0;font-size:11px;width:130px;flex-shrink:0;">{lbl}</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:2px;'
                f'height:8px;overflow:hidden;">'
                f'<div style="width:{round(pct)}%;background:{clr};height:100%;'
                f'border-radius:2px;"></div></div>'
                f'<span style="color:#94a3b8;font-size:11px;min-width:28px;text-align:right;">{cnt}</span>'
                f'<span style="color:rgba(255,255,255,0.3);font-size:10px;'
                f'min-width:38px;text-align:right;">{pct}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if r.get("description"):
        st.markdown(
            f'<div class="narrative-box" style="margin-top:12px;">'
            f'<div class="narrative-title">AI Narrative</div>'
            f'{str(r["description"])}'
            f'<div style="margin-top:6px;font-size:10px;color:rgba(255,255,255,0.3);">'
            f'Data year: {r["data_year"]} · Reliability: {r["reliability"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def render_grocery_diversity(
    domain_data: dict,
    map_data: dict = None,     # kept for backward compat, no longer used
    hood_filter=None,
):
    """
    Parameters
    ----------
    domain_data  : dict from load_domain("grocery", hood_filter)
    map_data     : ignored — geometry now comes from domain_data["map"]
    hood_filter  : str | None — sidebar neighbourhood selection
    """
    scores:  list = domain_data.get("scores",  [])
    summary: dict = domain_data.get("summary", {})

    food_deserts = summary.get("low_access_count", summary.get("food_desert_count", 0))
    if food_deserts:
        st.warning(f"⚠️ {food_deserts} neighborhoods classified as low access / food desert")

    if not scores:
        st.info("Grocery data not available.")
        return

    df       = _build_df(scores)
    df_chart = df[df["total_stores"] > 0].copy()

    if df.empty:
        st.info("Grocery data could not be prepared.")
        return

    if hood_filter:
        # ── Neighborhood selected: deep-dive hero first, then map + scorecard ──
        _section_deep_dive(df, hood_filter)
        _section_map(domain_data, df_chart, hood_filter)
        _section_scorecard(df)
    else:
        # ── All neighborhoods: KPI cards → bar/donut → map → scorecard ─────────
        _kpi_cards(df, summary)
        _section_overview(df)
        _section_map(domain_data, df_chart, hood_filter)
        _section_scorecard(df)