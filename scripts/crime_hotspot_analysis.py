"""
NeighbourWise AI — Crime Hotspot Analysis & Forecasting
Based on methodologies from:
  - Singh et al. (2025): Crime Pulse — DBSCAN spatial clustering
  - Cesario (2023) / Catlett et al. (2019): Spatio-temporal ARIMA forecasting

Improvement over both papers: Pre-defined neighborhood polygons (MASTER_LOCATION)
replace DBSCAN clustering — boundaries are semantically meaningful, not just
density-derived. SARIMAX forecasting with seasonal component + Snowflake Cortex
narrative generation added as Gen AI layer.

Run on Mac directly (NOT inside Docker).
"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
SF_CONFIG = {
    "account":       "pgb87192",
    "user":          os.environ["SNOWFLAKE_USER"],
    "password":      os.environ["SNOWFLAKE_PASSWORD"],
    "warehouse":     "NEIGHBOURWISE_AI",
    "database":      "NEIGHBOURWISE_DOMAINS",
    "schema":        "CRIME_ANALYSIS",
    "role":          "TRAINING_ROLE",
    "insecure_mode": True,
}

# SARIMAX via pmdarima auto_arima (stepwise)
MIN_MONTHS          = 12   # min time series length to fit
MIN_MONTHS_SEASONAL = 24   # need 2+ full cycles for seasonal component
FORECAST_HORIZON    = 6    # months ahead to forecast

# DBSCAN — haversine-aware params
EPS_KM          = 0.075  # 75m radius (~1 city block) — for stats
EPS_KM_MAP      = 0.150  # 150m radius — for map visualization
MIN_SAMPLES     = 20     # for stats 
MIN_SAMPLES_MAP = 120    # for map 


# ── SNOWFLAKE CONNECTION ──────────────────────────────────────────────────────
def get_conn():
    return snowflake.connector.connect(**SF_CONFIG)


# ── STEP 1: PULL CRIME DATA ───────────────────────────────────────────────────
def load_crime_data(conn) -> pd.DataFrame:
    """
    Pull directly from MRT_BOSTON_CRIME which already has NEIGHBORHOOD_NAME
    resolved via spatial join. No need to re-run ST_CONTAINS here.
    Filters to complete months only to avoid partial-month distortion.
    """
    query = """
    SELECT
        INCIDENT_NUMBER                         AS INCIDENT_ID,
        CITY,
        OFFENSE_DESCRIPTION                     AS CRIME_DESCRIPTION,
        OCCURRED_ON_DATE::DATE                  AS CRIME_DATE,
        DATE_TRUNC('month', OCCURRED_ON_DATE)   AS YEAR_MONTH,
        YEAR(OCCURRED_ON_DATE)                  AS CRIME_YEAR,
        MONTH(OCCURRED_ON_DATE)                 AS CRIME_MONTH,
        LAT,
        LONG,
        LOCATION_ID,
        NEIGHBORHOOD_NAME
    FROM NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_CRIME
    WHERE VALID_LOCATION = TRUE
      AND OCCURRED_ON_DATE IS NOT NULL
      AND NEIGHBORHOOD_NAME IS NOT NULL
      AND NEIGHBORHOOD_NAME != 'UNKNOWN'
      AND YEAR(OCCURRED_ON_DATE) >= 2023
      AND DATE_TRUNC('month', OCCURRED_ON_DATE) < DATE_TRUNC('month', CURRENT_DATE())
    """
    print("Loading crime data from MRT_BOSTON_CRIME...")
    df = pd.read_sql(query, conn)
    df.columns = [c.upper() for c in df.columns]
    print(f"  {len(df):,} crimes loaded across {df['NEIGHBORHOOD_NAME'].nunique()} neighborhoods")
    return df


# ── STEP 2: BUILD TIME SERIES PER NEIGHBORHOOD ───────────────────────────────
def build_time_series(df: pd.DataFrame) -> dict:
    df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
    monthly = (
        df.groupby(["NEIGHBORHOOD_NAME", "YEAR_MONTH"])
          .size()
          .reset_index(name="crime_count")
    )

    ts_dict = {}
    for nbhd, grp in monthly.groupby("NEIGHBORHOOD_NAME"):
        s = grp.set_index("YEAR_MONTH")["crime_count"].sort_index()
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        s = s.reindex(full_idx, fill_value=0)
        ts_dict[nbhd] = s

    print(f"  Built time series for {len(ts_dict)} neighborhoods")
    return ts_dict


# ── STEP 3: SARIMAX FORECASTING ──────────────────────────────────────────────
def check_stationarity(series: pd.Series) -> bool:
    try:
        return adfuller(series.dropna())[1] < 0.05
    except Exception:
        return False


def best_sarimax(series: pd.Series):
    """
    Use pmdarima auto_arima with stepwise search for fast SARIMAX fitting.
    """
    use_seasonal = len(series) >= MIN_MONTHS_SEASONAL
    
    try:
        model = auto_arima(
            series,
            seasonal=use_seasonal,
            m=12 if use_seasonal else 1,
            stepwise=True,           
            suppress_warnings=True,
            error_action="ignore",   
            max_p=2, max_q=2,        
            max_P=1, max_Q=1,        
            max_d=1, D=1,            # <--- FIX: D=1 applied here to force seasonal curves
            information_criterion="aic",
            n_fits=50,               
        )
        
        order = model.order
        seasonal_order = model.seasonal_order
        seasonal_str = f"({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})"
        print(f"    Best: SARIMAX{order}{seasonal_str} — AIC={model.aic():.1f}")
        
        return model, order, seasonal_order
        
    except Exception as e:
        print(f"    auto_arima failed: {e}")
        return None, None, None


def forecast_all_neighborhoods(ts_dict: dict) -> pd.DataFrame:
    results = []
    skipped = []

    for nbhd, series in ts_dict.items():
        if len(series) < MIN_MONTHS:
            skipped.append(nbhd)
            continue

        print(f"  Fitting {nbhd} ({len(series)} months)...")
        model, order, seasonal_order = best_sarimax(series)
        if model is None:
            skipped.append(nbhd)
            continue

        # ── Train/test MAPE calculation ───────────────────────────────
        try:
            train_s = series.iloc[:-6]
            test_s  = series.iloc[-6:]
            if len(train_s) >= MIN_MONTHS:
                use_seasonal_test = len(train_s) >= MIN_MONTHS_SEASONAL
                m_test = auto_arima(
                    train_s,
                    seasonal=use_seasonal_test,
                    m=12 if use_seasonal_test else 1,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=2, max_q=2,
                    max_P=1, max_Q=1,
                    max_d=1, D=1,    # <--- FIX: D=1 applied here for accurate MAPE testing
                    n_fits=50,
                )
                preds = m_test.predict(n_periods=6)
                mape  = np.mean(
                    np.abs((test_s.values - preds) / (test_s.values + 1e-9))
                ) * 100
            else:
                mape = None
        except Exception:
            mape = None

        # ── Generate forecast from full model ─────────────────────────
        fc_vals, ci = model.predict(n_periods=FORECAST_HORIZON, return_conf_int=True)
        
        # Build forecast dates starting from month after last training date
        last_date = series.index[-1]
        fc_dates  = pd.date_range(last_date + pd.DateOffset(months=1),
                                  periods=FORECAST_HORIZON, freq="MS")

        order_str    = str(order)
        seasonal_str = str((seasonal_order[0], seasonal_order[1],
                           seasonal_order[2], seasonal_order[3]))

        for i in range(FORECAST_HORIZON):
            results.append({
                "NEIGHBORHOOD_NAME": nbhd,
                "LAST_TRAIN_DATE":   last_date.strftime("%Y-%m"),
                "FORECAST_MONTH":    fc_dates[i].strftime("%Y-%m"),
                "FORECASTED_COUNT":  max(0, round(fc_vals[i])),
                "LOWER_CI":          max(0, round(ci[i, 0])),
                "UPPER_CI":          max(0, round(ci[i, 1])),
                "ARIMA_ORDER":       f"{order_str}{seasonal_str}",
                "TRAIN_MAPE":        round(min(mape, 999.0), 2) if mape is not None else None,
            })

    if skipped:
        print(f"  Skipped {len(skipped)} neighborhoods (insufficient data): {skipped}")

    df = pd.DataFrame(results)
    print(f"  Forecasted {df['NEIGHBORHOOD_NAME'].nunique()} neighborhoods")
    return df


# ── STEP 4: DBSCAN HOTSPOT ANALYSIS ──────────────────────────────────────────
def dbscan_hotspot_analysis(df: pd.DataFrame):
    """
    Two-pass DBSCAN:
      Pass 1 (75m):  Fine-grained stats for CA_CRIME_HOTSPOT_CLUSTERS
      Pass 2 (150m): Broader zones for CA_CRIME_CLUSTER_POINTS (map viz)
    
    Both use last 12 months only.
    Returns: (summary_df, points_df)
    """
    summary_results = []
    point_results   = []
    eps_rad_stats   = EPS_KM / 6371.0       # 75m for stats
    eps_rad_map     = EPS_KM_MAP / 6371.0    # 150m for map

    df["CRIME_DATE"] = pd.to_datetime(df["CRIME_DATE"])
    cutoff = df["CRIME_DATE"].max() - pd.DateOffset(months=12)
    df_recent = df[df["CRIME_DATE"] >= cutoff].copy()
    print(f"  DBSCAN using last 12 months: {len(df_recent):,} crimes (of {len(df):,} total)")

    for nbhd, grp in df_recent.groupby("NEIGHBORHOOD_NAME"):
        coords = grp[["LAT", "LONG"]].dropna()
        if len(coords) < MIN_SAMPLES:
            continue

        coords_vals = coords.values
        coords_rad  = np.radians(coords_vals)

        # ── Pass 1: Fine-grained (75m) for stats ──────────────────────
        db_stats    = DBSCAN(eps=eps_rad_stats, min_samples=MIN_SAMPLES,
                             metric="haversine").fit(coords_rad)
        labels_stats = db_stats.labels_

        n_clusters  = len(set(labels_stats)) - (1 if -1 in labels_stats else 0)
        noise_pct   = (labels_stats == -1).sum() / len(labels_stats) * 100
        hotspot_pct = 100 - noise_pct

        summary_results.append({
            "NEIGHBORHOOD_NAME":        nbhd,
            "TOTAL_CRIMES":             len(coords_vals),
            "N_HOTSPOT_CLUSTERS":       n_clusters,
            "HOTSPOT_CRIME_SHARE_PCT":  round(hotspot_pct, 1),
            "NOISE_CRIME_PCT":          round(noise_pct, 1),
        })

        # ── Pass 2: Map visualization (150m) ─────────
        db_map     = DBSCAN(eps=eps_rad_map, min_samples=MIN_SAMPLES_MAP,
                            metric="haversine").fit(coords_rad)
        labels_map = db_map.labels_

        grp_filtered = grp.loc[coords.index].copy()
        grp_filtered["CLUSTER_ID"] = labels_map.astype(int)
        grp_filtered["IS_NOISE"]   = labels_map == -1

        for _, row in grp_filtered.iterrows():
            point_results.append({
                "NEIGHBORHOOD_NAME": nbhd,
                "LAT":               row["LAT"],
                "LONG":              row["LONG"],
                "CLUSTER_ID":        int(row["CLUSTER_ID"]),
                "IS_NOISE":          bool(row["IS_NOISE"]),
                "CRIME_DATE":        row["CRIME_DATE"].strftime("%Y-%m-%d") if pd.notna(row["CRIME_DATE"]) else None,
                "CRIME_DESCRIPTION": row.get("CRIME_DESCRIPTION", None),
            })

    summary_df = pd.DataFrame(summary_results).sort_values(
        "HOTSPOT_CRIME_SHARE_PCT", ascending=False
    )
    points_df = pd.DataFrame(point_results)

    print(f"  DBSCAN stats (75m): {len(summary_df)} neighborhoods")
    print(f"  DBSCAN map (150m): {len(points_df):,} crime points with cluster labels")
    return summary_df, points_df


# ── STEP 5: CORTEX NARRATIVE GENERATION ──────────────────────────────────────
def generate_cortex_narratives(conn, forecast_df: pd.DataFrame,
                                dbscan_df: pd.DataFrame,
                                ts_dict: dict) -> pd.DataFrame:
    narratives   = []
    dbscan_lookup = dbscan_df.set_index("NEIGHBORHOOD_NAME").to_dict("index")

    first_fc = (
        forecast_df.sort_values("FORECAST_MONTH")
                   .groupby("NEIGHBORHOOD_NAME")
                   .first()
                   .reset_index()
    )

    for _, row in first_fc.iterrows():
        nbhd = row["NEIGHBORHOOD_NAME"]
        ts   = ts_dict.get(nbhd)
        if ts is None or len(ts) < 3:
            continue

        recent_avg = ts.iloc[-3:].mean()
        
        # YoY comparison
        if len(ts) >= 15:
            prev_year_avg = ts.iloc[-15:-12].mean()
            
            if prev_year_avg == 0:
                trend_dir = "increasing" if recent_avg > 0 else "stable"
            else:
                trend_dir  = (
                    "increasing" if recent_avg > prev_year_avg * 1.05 else
                    "decreasing" if recent_avg < prev_year_avg * 0.95 else
                    "stable"
                )
        else:
            trend_dir = "stable" 
            prev_year_avg = recent_avg

        fc_val      = row["FORECASTED_COUNT"]
        fc_month    = row["FORECAST_MONTH"]
        mape        = row["TRAIN_MAPE"]
        dbscan_info = dbscan_lookup.get(nbhd, {})
        n_clusters  = dbscan_info.get("N_HOTSPOT_CLUSTERS", "unknown")
        hotspot_pct = dbscan_info.get("HOTSPOT_CRIME_SHARE_PCT", "unknown")

        prompt = (
            f"You are a neighborhood safety analyst writing for a Boston/Cambridge, "
            f"Massachusetts neighborhood guide. "
            f"Write a 2-3 sentence safety trend summary for the {nbhd} neighborhood "
            f"in Massachusetts. "
            f"IMPORTANT: {nbhd} is a neighborhood in the Boston/Cambridge, Massachusetts area. "
            f"Do not confuse it with any place of the same name elsewhere.\n\n"
            f"Data:\n"
            f"- Recent 3-month average monthly crimes: {recent_avg:.0f}\n"
            f"- Same 3-month period last year average: {prev_year_avg:.0f}\n"
            f"- Year-Over-Year recent trend: {trend_dir}\n"
            f"- ARIMA forecast for {fc_month}: {fc_val} crimes "
            f"(model accuracy MAPE: {mape}%)\n"
            f"- Number of extreme crime hotspot clusters within neighborhood: {n_clusters}\n"
            f"- Percentage of crimes occurring in these dense clusters: {hotspot_pct}%\n\n"
            f"Write a clear, factual, non-alarmist summary suitable for a neighborhood "
            f"recommendation app. Do not use bullet points. Keep it under 60 words."
        )

        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', %s)", (prompt,)
            )
            narrative = cur.fetchone()[0].strip()
            cur.close()
        except Exception as e:
            narrative = f"Safety trend data available. Recent crime trend is {trend_dir}."
            print(f"  Cortex error for {nbhd}: {e}")

        narratives.append({
            "NEIGHBORHOOD_NAME":  nbhd,
            "RECENT_TREND":       trend_dir,
            "RECENT_AVG_MONTHLY": round(float(recent_avg), 1),
            "FORECAST_MONTH":     fc_month,
            "FORECASTED_COUNT":   int(fc_val),
            "TRAIN_MAPE":         float(mape) if mape is not None else None,
            "N_HOTSPOT_CLUSTERS": int(n_clusters) if n_clusters != "unknown" else None,
            "SAFETY_NARRATIVE":   narrative,
            "RELIABILITY_FLAG":   "LOW" if (
                (mape is not None and mape > 50) or
                len(ts_dict.get(nbhd, [])) < 12 or
                ts_dict.get(nbhd, pd.Series()).sum() < 500
            ) else "HIGH",
        })

    df_out = pd.DataFrame(narratives)
    print(f"  Generated {len(df_out)} Cortex narratives")
    return df_out


# ── STEP 6: WRITE RESULTS TO SNOWFLAKE ───────────────────────────────────────
def _create_tables(cur):
    """Create or replace all 4 output tables."""
    cur.execute("""
    CREATE OR REPLACE TABLE NEIGHBOURWISE_DOMAINS.CRIME_ANALYSIS.CA_CRIME_FORECAST (
        NEIGHBORHOOD_NAME  VARCHAR(100),
        LAST_TRAIN_DATE    VARCHAR(10),
        FORECAST_MONTH     VARCHAR(10),
        FORECASTED_COUNT   INTEGER,
        LOWER_CI           INTEGER,
        UPPER_CI           INTEGER,
        ARIMA_ORDER        VARCHAR(40),
        TRAIN_MAPE         FLOAT,
        LOAD_TIMESTAMP     TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )""")

    cur.execute("""
    CREATE OR REPLACE TABLE NEIGHBOURWISE_DOMAINS.CRIME_ANALYSIS.CA_CRIME_HOTSPOT_CLUSTERS (
        NEIGHBORHOOD_NAME        VARCHAR(100),
        TOTAL_CRIMES             INTEGER,
        N_HOTSPOT_CLUSTERS       INTEGER,
        HOTSPOT_CRIME_SHARE_PCT  FLOAT,
        NOISE_CRIME_PCT          FLOAT,
        LOAD_TIMESTAMP           TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )""")

    cur.execute("""
    CREATE OR REPLACE TABLE NEIGHBOURWISE_DOMAINS.CRIME_ANALYSIS.CA_CRIME_SAFETY_NARRATIVE (
        NEIGHBORHOOD_NAME   VARCHAR(100),
        RECENT_TREND        VARCHAR(20),
        RECENT_AVG_MONTHLY  FLOAT,
        FORECAST_MONTH      VARCHAR(10),
        FORECASTED_COUNT    INTEGER,
        TRAIN_MAPE          FLOAT,
        N_HOTSPOT_CLUSTERS  INTEGER,
        SAFETY_NARRATIVE    VARCHAR(2000),
        RELIABILITY_FLAG    VARCHAR(10),
        LOAD_TIMESTAMP      TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )""")

    cur.execute("""
    CREATE OR REPLACE TABLE NEIGHBOURWISE_DOMAINS.CRIME_ANALYSIS.CA_CRIME_CLUSTER_POINTS (
        NEIGHBORHOOD_NAME   VARCHAR(100),
        LAT                 FLOAT,
        LONG                FLOAT,
        CLUSTER_ID          INTEGER,
        IS_NOISE            BOOLEAN,
        CRIME_DATE          VARCHAR(10),
        CRIME_DESCRIPTION   VARCHAR(500),
        LOAD_TIMESTAMP      TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )""")


def write_results(conn, forecast_df: pd.DataFrame,
                  dbscan_df: pd.DataFrame,
                  narrative_df: pd.DataFrame,
                  cluster_points_df: pd.DataFrame):
    cur = conn.cursor()
    _create_tables(cur)
    cur.close()

    def _upload(df, table):
        success, nchunks, nrows, _ = write_pandas(
            conn, df, table,
            database="NEIGHBOURWISE_DOMAINS",
            schema="CRIME_ANALYSIS",
            auto_create_table=False,
            overwrite=False,
        )
        print(f"  Wrote {nrows:,} rows to {table} ({nchunks} chunk(s))")

    _upload(forecast_df,        "CA_CRIME_FORECAST")
    _upload(dbscan_df,          "CA_CRIME_HOTSPOT_CLUSTERS")
    _upload(narrative_df,       "CA_CRIME_SAFETY_NARRATIVE")
    _upload(cluster_points_df,  "CA_CRIME_CLUSTER_POINTS")

    conn.commit()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("NeighbourWise Crime Hotspot Analysis & Forecasting")
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    conn = get_conn()

    try:
        print("STEP 1 — Loading crime data from MRT_BOSTON_CRIME")
        df = load_crime_data(conn)

        print("\nSTEP 2 — Building monthly time series per neighborhood")
        ts_dict = build_time_series(df)

        print("\nSTEP 3 — SARIMAX forecasting per neighborhood")
        forecast_df = forecast_all_neighborhoods(ts_dict)

        print("\nSTEP 4 — DBSCAN sub-neighborhood hotspot clustering")
        dbscan_df, cluster_points_df = dbscan_hotspot_analysis(df)

        print("\nSTEP 5 — Generating Cortex safety narratives")
        narrative_df = generate_cortex_narratives(conn, forecast_df, dbscan_df, ts_dict)

        print("\nSTEP 6 — Writing results to Snowflake CRIME_ANALYSIS")
        write_results(conn, forecast_df, dbscan_df, narrative_df, cluster_points_df)

        print("\n" + "="*60)
        print("SAMPLE OUTPUT — Safety Narratives")
        print("="*60)
        for _, r in narrative_df.head(5).iterrows():
            print(f"\n{r['NEIGHBORHOOD_NAME']} ({r['RECENT_TREND']} trend)")
            print(f"  Forecast ({r['FORECAST_MONTH']}): {r['FORECASTED_COUNT']} crimes")
            print(f"  Narrative: {r['SAFETY_NARRATIVE']}")

        print(f"\n{'='*60}")
        print("Run completed successfully.")
        print("Tables in CRIME_ANALYSIS schema:")
        print("  CA_CRIME_FORECAST")
        print("  CA_CRIME_HOTSPOT_CLUSTERS")
        print("  CA_CRIME_SAFETY_NARRATIVE")
        print("  CA_CRIME_CLUSTER_POINTS")
        print(f"  Cluster points: {len(cluster_points_df):,} rows")

    finally:
        conn.close()


if __name__ == "__main__":
    main()