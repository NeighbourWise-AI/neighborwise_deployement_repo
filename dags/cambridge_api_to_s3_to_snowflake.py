from __future__ import annotations

import os
import csv
import time
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import snowflake.connector

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
}

CAMBRIDGE_API_URL = "https://data.cambridgema.gov/resource/xuad-73uj.json"
PAGE_SIZE         = 10000


def _get_resilient_session():
    """Create a requests session with automatic retry on transient failures."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ── Task 1: Fetch from Cambridge SODA2 API → CSV ──────────────────────────────
def fetch_and_convert_to_csv(**context):
    """
    Paginate through Cambridge Open Data SODA2 API and write to CSV.
    Resilient session + per-page retry for connection-level errors.
    """
    ts       = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    csv_path = f"{base_dir}/cambridge_crime_{ts}.csv"

    session     = _get_resilient_session()
    all_records = []
    offset      = 0
    max_page_retries = 3

    print(f"Fetching Cambridge crime data from {CAMBRIDGE_API_URL}")

    while True:
        batch = None
        for attempt in range(1, max_page_retries + 1):
            try:
                resp = session.get(
                    CAMBRIDGE_API_URL,
                    params={
                        "$limit":  PAGE_SIZE,
                        "$offset": offset,
                        "$order":  "file_number ASC",
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                batch = resp.json()
                break
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                wait = 2 ** attempt
                print(f"  Page fetch error (attempt {attempt}/{max_page_retries}): {e}")
                if attempt == max_page_retries:
                    print(f"  All retries failed at offset {offset}. "
                          f"{len(all_records):,} records fetched before failure.")
                    raise
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

        if not batch:
            break

        all_records.extend(batch)
        print(f"Fetched {len(all_records):,} records so far...")
        offset += PAGE_SIZE

        if len(batch) < PAGE_SIZE:
            break

    print(f"Total records fetched: {len(all_records):,}")

    if not all_records:
        raise ValueError("No records returned from Cambridge API")

    fieldnames = [
        "file_number", "date_of_report", "crime_date_time",
        "crime", "reporting_area", "neighborhood",
        "location", "reporting_area_lat", "reporting_area_lon",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in all_records:
            writer.writerow({
                "file_number":        record.get("file_number", ""),
                "date_of_report":     record.get("date_of_report", ""),
                "crime_date_time":    record.get("crime_date_time", ""),
                "crime":              record.get("crime", ""),
                "reporting_area":     record.get("reporting_area", ""),
                "neighborhood":       record.get("neighborhood", ""),
                "location":           record.get("location", ""),
                "reporting_area_lat": record.get("reporting_area_lat", ""),
                "reporting_area_lon": record.get("reporting_area_lon", ""),
            })

    print(f"CSV written: {csv_path}")

    context["task_instance"].xcom_push(key="csv_path",      value=csv_path)
    context["task_instance"].xcom_push(key="timestamp",     value=ts)
    context["task_instance"].xcom_push(key="record_count",  value=len(all_records))

    return {"csv_path": csv_path, "record_count": len(all_records)}


# ── Task 2: Upload CSV → S3 ────────────────────────────────────────────────────
def upload_to_s3(**context):
    """Upload CSV to S3 using S3Hook."""
    ti          = context["task_instance"]
    csv_path    = ti.xcom_pull(task_ids="fetch_and_convert", key="csv_path")
    ts          = ti.xcom_pull(task_ids="fetch_and_convert", key="timestamp")

    s3_bucket   = Variable.get("cambridge_s3_bucket",     default_var=Variable.get("boston_s3_bucket"))
    s3_prefix   = Variable.get("cambridge_s3_key_prefix", default_var="crime-safety/")
    aws_conn_id = Variable.get("boston_aws_conn_id",       default_var="aws_default")

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    s3_key = f"{s3_prefix}cambridge_crime_{ts}.csv"
    print(f"Uploading to s3://{s3_bucket}/{s3_key}")

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(
        filename=csv_path,
        key=s3_key,
        bucket_name=s3_bucket,
        replace=True,
    )

    print(f"Upload complete: s3://{s3_bucket}/{s3_key}")

    ti.xcom_push(key="s3_key",    value=s3_key)
    ti.xcom_push(key="s3_bucket", value=s3_bucket)

    return s3_key


# ── Task 3: Create Snowflake staging table ─────────────────────────────────────
def create_snowflake_table(**context):
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True,
    )
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_CAMBRIDGE_CRIME (
                FILE_NUMBER         VARCHAR(50),
                DATE_OF_REPORT      VARCHAR(50),
                CRIME_DATE_TIME     VARCHAR(50),
                CRIME               VARCHAR(200),
                REPORTING_AREA      VARCHAR(20),
                NEIGHBORHOOD        VARCHAR(100),
                LOCATION            VARCHAR(300),
                REPORTING_AREA_LAT  FLOAT,
                REPORTING_AREA_LON  FLOAT,
                LOAD_TIMESTAMP      TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        print("Table STG_CAMBRIDGE_CRIME created or already exists")
    finally:
        cur.close()
        conn.close()


# ── Task 4: COPY from S3 → Snowflake ──────────────────────────────────────────
def load_s3_to_snowflake(**context):
    """TRUNCATE + reload — full weekly pull."""
    ti        = context["task_instance"]
    s3_bucket = ti.xcom_pull(task_ids="upload_to_s3", key="s3_bucket")
    s3_key    = ti.xcom_pull(task_ids="upload_to_s3", key="s3_key")
    s3_path   = f"s3://{s3_bucket}/{s3_key}"

    from airflow.hooks.base import BaseHook
    aws_conn       = BaseHook.get_connection("aws_default")
    aws_access_key = aws_conn.login
    aws_secret_key = aws_conn.password

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True,
    )
    try:
        cur = conn.cursor()

        cur.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_CAMBRIDGE_CRIME")
        print("Table truncated")

        cur.execute(f"""
            COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_CAMBRIDGE_CRIME
                (FILE_NUMBER, DATE_OF_REPORT, CRIME_DATE_TIME, CRIME,
                 REPORTING_AREA, NEIGHBORHOOD, LOCATION,
                 REPORTING_AREA_LAT, REPORTING_AREA_LON)
            FROM '{s3_path}'
            CREDENTIALS = (
                AWS_KEY_ID     = '{aws_access_key}'
                AWS_SECRET_KEY = '{aws_secret_key}'
            )
            FILE_FORMAT = (
                TYPE                         = 'CSV'
                FIELD_DELIMITER              = ','
                SKIP_HEADER                  = 1
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                TRIM_SPACE                   = TRUE
                ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
                NULL_IF                      = ('NULL', 'null', '')
                EMPTY_FIELD_AS_NULL          = TRUE
            )
            ON_ERROR = 'CONTINUE'
            PURGE    = FALSE
        """)

        cur.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_CAMBRIDGE_CRIME")
        count = cur.fetchone()[0]
        print(f"Total records in STG_CAMBRIDGE_CRIME: {count:,}")

        return {"status": "success", "records_loaded": count}

    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    finally:
        cur.close()
        conn.close()


# ── DAG Definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="cambridge_crime_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    description="Cambridge PD crime incidents: SODA2 API → S3 → Snowflake STG_CAMBRIDGE_CRIME",
    start_date=datetime(2026, 2, 28),
    schedule_interval="0 0 * * 0",
    catchup=False,
    tags=["cambridge", "crime", "neighbourwise"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_and_convert",
        python_callable=fetch_and_convert_to_csv,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id="create_snowflake_table",
        python_callable=create_snowflake_table,
        provide_context=True,
    )

    t4 = PythonOperator(
        task_id="load_to_snowflake",
        python_callable=load_s3_to_snowflake,
        provide_context=True,
    )

    t1 >> t2 >> t3 >> t4