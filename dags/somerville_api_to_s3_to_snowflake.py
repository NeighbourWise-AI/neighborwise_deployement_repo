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

# Somerville Open Data Portal — Socrata API
# Dataset: Police Data: Crime Reports (aghs-hqvg)
# Docs: https://data.somervillema.gov/Public-Safety/Police-Data-Crime-Reports/aghs-hqvg
SOMERVILLE_API_BASE = "https://data.somervillema.gov/resource/aghs-hqvg.json"


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


def fetch_and_convert_to_csv(**context):
    """Fetch Somerville crime data from Socrata API, paginate, save as CSV.

    Socrata API uses $limit and $offset for pagination.
    App token optional but recommended to avoid throttling.
    """
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    csv_path = f"{base_dir}/somerville_crime_{ts}.csv"

    # Optional: set somerville_socrata_app_token in Airflow Variables for higher rate limits
    app_token = Variable.get("somerville_socrata_app_token", default_var=None)

    session = _get_resilient_session()
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    all_records = []
    offset = 0
    limit = 10000
    max_page_retries = 3

    print(f"Fetching Somerville crime data from Socrata API")

    while True:
        url = f"{SOMERVILLE_API_BASE}?$limit={limit}&$offset={offset}&$order=:id"
        print(f"  Fetching records {offset} to {offset + limit}...")

        records = None
        for attempt in range(1, max_page_retries + 1):
            try:
                resp = session.get(url, headers=headers, timeout=120)
                resp.raise_for_status()
                records = resp.json()
                break
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                wait = 2 ** attempt
                print(f"    Page fetch error (attempt {attempt}/{max_page_retries}): {e}")
                if attempt == max_page_retries:
                    print(f"    All retries failed at offset {offset}. "
                          f"Saving {len(all_records)} records fetched so far.")
                    raise
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)

        if not records:
            print("  No more records found")
            break

        all_records.extend(records)
        print(f"  Fetched {len(records)} records. Total so far: {len(all_records)}")

        if len(records) < limit:
            print("  Reached last page")
            break

        offset += limit

    print(f"Finished fetching. Total records: {len(all_records)}")
    if not all_records:
        raise ValueError("No records found in Somerville API response")

    # Normalize keys — Socrata returns lowercase; write CSV with uppercase for Snowflake
    # Actual Socrata fields from: data.somervillema.gov/resource/aghs-hqvg.json
    FIELD_MAP = {
        "incnum":         "INCIDENT_NUMBER",
        "day_and_month":  "DAY_AND_MONTH",
        "year":           "YEAR",
        "police_shift":   "POLICE_SHIFT",
        "offensecode":    "OFFENSE_CODE",
        "offense":        "OFFENSE",
        "incdesc":        "OFFENSE_DESCRIPTION",
        "offensetype":    "OFFENSE_TYPE",         # e.g. "Fraud Offenses", "Assault Offenses"
        "category":       "CRIME_CATEGORY",       # Crimes Against Person/Property/Society
        "blockcode":      "BLOCK_CODE",           # Census block code (no lat/long available)
        "ward":           "WARD",
    }

    fieldnames = list(FIELD_MAP.values())

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            row = {}
            for src_key, dest_key in FIELD_MAP.items():
                row[dest_key] = rec.get(src_key, "")
            writer.writerow(row)

    print(f"CSV saved to {csv_path} ({len(all_records)} rows)")

    context['task_instance'].xcom_push(key='csv_path', value=csv_path)
    context['task_instance'].xcom_push(key='timestamp', value=ts)

    return {"csv_path": csv_path, "record_count": len(all_records)}


def upload_to_s3(**context):
    """Upload the CSV file to S3."""
    s3_bucket = Variable.get("boston_s3_bucket")
    s3_prefix = Variable.get("boston_s3_key_prefix", default_var="crime-safety/")
    aws_conn_id = Variable.get("boston_aws_conn_id", default_var="aws_default")

    ti = context['task_instance']
    csv_path = ti.xcom_pull(task_ids='fetch_and_convert', key='csv_path')
    ts = ti.xcom_pull(task_ids='fetch_and_convert', key='timestamp')

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    s3_key = f"{s3_prefix}somerville_crime_{ts}.csv"
    print(f"Uploading {csv_path} to s3://{s3_bucket}/{s3_key}")

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(filename=csv_path, key=s3_key, bucket_name=s3_bucket, replace=True)

    print(f"Successfully uploaded to S3: s3://{s3_bucket}/{s3_key}")
    context['task_instance'].xcom_push(key='s3_key', value=s3_key)
    context['task_instance'].xcom_push(key='s3_bucket', value=s3_bucket)

    return s3_key


def create_snowflake_table(**context):
    """Create Snowflake staging table for Somerville crime data."""
    conn = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        schema=os.environ['SNOWFLAKE_SCHEMA'],
        role=os.environ['SNOWFLAKE_ROLE'],
        insecure_mode=True
    )

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_SOMERVILLE_CRIME (
        INCIDENT_NUMBER VARCHAR(50),
        DAY_AND_MONTH VARCHAR(20),
        YEAR NUMBER,
        POLICE_SHIFT VARCHAR(50),
        OFFENSE_CODE VARCHAR(20),
        OFFENSE VARCHAR(200),
        OFFENSE_DESCRIPTION VARCHAR(1000),
        OFFENSE_TYPE VARCHAR(100),
        CRIME_CATEGORY VARCHAR(100),
        BLOCK_CODE VARCHAR(50),
        WARD VARCHAR(20),
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """

    try:
        cursor = conn.cursor()
        print("Creating Snowflake table STG_SOMERVILLE_CRIME")
        cursor.execute(create_table_sql)
        print("Table STG_SOMERVILLE_CRIME created or already exists")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()


def load_s3_to_snowflake(**context):
    """Load data from S3 to Snowflake using COPY INTO."""
    ti = context['task_instance']
    s3_bucket = ti.xcom_pull(task_ids='upload_to_s3', key='s3_bucket')
    s3_key = ti.xcom_pull(task_ids='upload_to_s3', key='s3_key')

    from airflow.hooks.base import BaseHook
    aws_conn = BaseHook.get_connection('aws_default')
    aws_access_key = aws_conn.login
    aws_secret_key = aws_conn.password

    s3_path = f"s3://{s3_bucket}/{s3_key}"

    conn = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        schema=os.environ['SNOWFLAKE_SCHEMA'],
        role=os.environ['SNOWFLAKE_ROLE'],
        insecure_mode=True
    )

    copy_sql = f"""
    COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_SOMERVILLE_CRIME
        (INCIDENT_NUMBER, DAY_AND_MONTH, YEAR, POLICE_SHIFT, OFFENSE_CODE,
         OFFENSE, OFFENSE_DESCRIPTION, OFFENSE_TYPE, CRIME_CATEGORY, BLOCK_CODE, WARD)
    FROM '{s3_path}'
    CREDENTIALS = (
        AWS_KEY_ID = '{aws_access_key}'
        AWS_SECRET_KEY = '{aws_secret_key}'
    )
    FILE_FORMAT = (
        TYPE = 'CSV'
        FIELD_DELIMITER = ','
        SKIP_HEADER = 1
        FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        TRIM_SPACE = TRUE
        ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
        NULL_IF = ('NULL', 'null', '')
    )
    ON_ERROR = 'CONTINUE'
    PURGE = FALSE;
    """

    try:
        cursor = conn.cursor()
        print("Truncating STG_SOMERVILLE_CRIME before fresh load")
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_SOMERVILLE_CRIME")
        print(f"Loading data from {s3_path} to Snowflake")
        cursor.execute(copy_sql)

        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_SOMERVILLE_CRIME")
        count = cursor.fetchone()[0]
        print(f"Total records in STG_SOMERVILLE_CRIME: {count}")

        return {"status": "success", "records_loaded": count}
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="somerville_crime_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 3, 16),
    schedule_interval='0 0 * * 0',    # Weekly Sunday midnight, same as Boston
    catchup=False,
    tags=["somerville", "crime", "s3", "api"],
) as dag:

    fetch_and_convert_task = PythonOperator(
        task_id="fetch_and_convert",
        python_callable=fetch_and_convert_to_csv,
        provide_context=True,
    )

    upload_task = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3,
        provide_context=True,
    )

    create_table_task = PythonOperator(
        task_id="create_snowflake_table",
        python_callable=create_snowflake_table,
        provide_context=True,
    )

    load_to_snowflake_task = PythonOperator(
        task_id="load_to_snowflake",
        python_callable=load_s3_to_snowflake,
        provide_context=True,
    )

    fetch_and_convert_task >> upload_task >> create_table_task >> load_to_snowflake_task