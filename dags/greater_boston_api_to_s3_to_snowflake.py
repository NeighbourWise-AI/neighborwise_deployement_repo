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

# FBI Crime Data Explorer API — summarized offense counts by agency ORI
# Free API key: https://api.data.gov/signup/
# Docs: https://crime-data-explorer.fr.cloud.gov/pages/docApi
#
# ORI codes for our 11 Greater Boston cities (excluding Somerville — handled separately)
# Format: MA + 5-digit agency code + 00
# These can be looked up at: https://api.usa.gov/crime/fbi/cde/agency/byStateAbbr/MA?API_KEY={key}
GREATER_BOSTON_AGENCIES = {
    "ARLINGTON":  "MA0090200",   # Middlesex — Arlington Police Department
    "BROOKLINE":  "MA0110400",   # Norfolk  — Brookline Police Department
    "NEWTON":     "MA0093300",   # Middlesex — Newton Police Department
    "WATERTOWN":  "MA0094800",   # Middlesex — Watertown Police Department
    "MEDFORD":    "MA0093000",   # Middlesex — Medford Police Department
    "MALDEN":     "MA0092700",   # Middlesex — Malden Police Department
    "REVERE":     "MA0130400",   # Suffolk  — Revere Police Department
    "CHELSEA":    "MA0130300",   # Suffolk  — Chelsea Police Department
    "EVERETT":    "MA0091700",   # Middlesex — Everett Police Department
    "SALEM":      "MA0052800",   # Essex    — Salem Police Department
    "QUINCY":     "MA0112000",   # Norfolk  — Quincy Police Department
}

# FBI CDE Summarized endpoint — verified offense slugs
# Endpoint: GET /summarized/agency/{ori}/{offense}?from=MM-YYYY&to=MM-YYYY&API_KEY={key}
# Valid offenses (from https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/docApi):
#   homicide, rape, robbery, aggravated-assault, burglary, larceny, motor-vehicle-theft, arson
OFFENSE_TYPES = {
    # HIGH severity (violent)
    "homicide":             "HIGH",
    "rape":                 "HIGH",
    "robbery":              "HIGH",
    "aggravated-assault":   "HIGH",
    # MEDIUM severity (property)
    "burglary":             "MEDIUM",
    "larceny":              "MEDIUM",
    "motor-vehicle-theft":  "MEDIUM",
    "arson":                "MEDIUM",
}

# Date range — MM-YYYY format required by API
DATE_FROM = "01-2022"
DATE_TO = "12-2024"


def _get_resilient_session():
    """Create a requests session with automatic retry on transient failures."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_and_convert_to_csv(**context):
    """Fetch aggregate crime data from FBI CDE Summarized API for all 11 Greater Boston cities.

    Verified API format:
      URL: https://api.usa.gov/crime/fbi/cde/summarized/agency/{ORI}/{offense}?from=01-2022&to=12-2024&API_KEY={key}
      Response: JSON with offenses.actuals."{Agency} Police Department Offenses" containing
                monthly counts keyed by "MM-YYYY" (e.g. {"01-2022": 4, "02-2022": 3, ...})
    """
    api_key = Variable.get("fbi_cde_api_key")
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    csv_path = f"{base_dir}/greater_boston_crime_{ts}.csv"

    session = _get_resilient_session()
    base_url = "https://api.usa.gov/crime/fbi/cde"

    all_rows = []
    max_retries = 3

    for city, ori in GREATER_BOSTON_AGENCIES.items():
        print(f"\nFetching data for {city} (ORI: {ori})")

        for offense, severity in OFFENSE_TYPES.items():
            url = (f"{base_url}/summarized/agency/{ori}/{offense}"
                   f"?from={DATE_FROM}&to={DATE_TO}&API_KEY={api_key}")

            data = None
            for attempt in range(1, max_retries + 1):
                try:
                    resp = session.get(url, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    elif resp.status_code == 404:
                        print(f"  {offense}: no data (404)")
                        break
                    else:
                        print(f"  {offense}: HTTP {resp.status_code} (attempt {attempt})")
                except (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout) as e:
                    print(f"  {offense}: error (attempt {attempt}): {e}")

                if attempt < max_retries:
                    time.sleep(2 ** attempt)

            if not data:
                continue

            # Parse response: offenses.actuals."{City} Police Department Offenses"
            # Keys are "MM-YYYY", values are integer counts
            try:
                actuals = data.get("offenses", {}).get("actuals", {})

                # Find the key ending in "Offenses" that matches our agency (not state/national)
                agency_offenses = None
                for key, val in actuals.items():
                    if key.endswith("Offenses") and "United States" not in key and "Massachusetts" not in key:
                        agency_offenses = val
                        break

                if not agency_offenses:
                    print(f"  {offense}: no agency-level actuals found in response")
                    continue

                # Aggregate monthly counts into annual totals
                annual_totals = {}
                for month_key, count in agency_offenses.items():
                    # month_key format: "MM-YYYY"
                    year = int(month_key.split("-")[1])
                    annual_totals[year] = annual_totals.get(year, 0) + int(count)

                for year, total in annual_totals.items():
                    all_rows.append({
                        "CITY":             city,
                        "ORI":              ori,
                        "YEAR":             year,
                        "OFFENSE_TYPE":     offense,
                        "SEVERITY":         severity,
                        "OFFENSE_COUNT":    total,
                        "DATA_SOURCE":      "FBI_CDE",
                    })

                print(f"  {offense}: {sum(annual_totals.values())} total across {len(annual_totals)} years")

            except Exception as e:
                print(f"  {offense}: parse error: {e}")
                continue

            time.sleep(0.5)  # Rate limit courtesy

    print(f"\nFinished fetching. Total rows: {len(all_rows)}")
    if not all_rows:
        raise ValueError("No records fetched from FBI CDE API. Check API key and ORI codes.")

    # Write CSV
    fieldnames = ["CITY", "ORI", "YEAR", "OFFENSE_TYPE", "SEVERITY", "OFFENSE_COUNT", "DATA_SOURCE"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"CSV saved to {csv_path} ({len(all_rows)} rows)")

    context['task_instance'].xcom_push(key='csv_path', value=csv_path)
    context['task_instance'].xcom_push(key='timestamp', value=ts)

    return {"csv_path": csv_path, "record_count": len(all_rows)}


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

    s3_key = f"{s3_prefix}greater_boston_crime_{ts}.csv"
    print(f"Uploading {csv_path} to s3://{s3_bucket}/{s3_key}")

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(filename=csv_path, key=s3_key, bucket_name=s3_bucket, replace=True)

    print(f"Successfully uploaded to S3: s3://{s3_bucket}/{s3_key}")
    context['task_instance'].xcom_push(key='s3_key', value=s3_key)
    context['task_instance'].xcom_push(key='s3_bucket', value=s3_bucket)

    return s3_key


def create_snowflake_table(**context):
    """Create Snowflake staging table for Greater Boston aggregate crime data."""
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
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_GREATER_BOSTON_CRIME (
        CITY VARCHAR(100),
        ORI VARCHAR(20),
        YEAR NUMBER,
        OFFENSE_TYPE VARCHAR(100),
        SEVERITY VARCHAR(20),
        OFFENSE_COUNT NUMBER,
        DATA_SOURCE VARCHAR(20),
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """

    try:
        cursor = conn.cursor()
        print("Creating Snowflake table STG_GREATER_BOSTON_CRIME")
        cursor.execute(create_table_sql)
        print("Table STG_GREATER_BOSTON_CRIME created or already exists")
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
    COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_GREATER_BOSTON_CRIME
        (CITY, ORI, YEAR, OFFENSE_TYPE, SEVERITY, OFFENSE_COUNT, DATA_SOURCE)
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
        print("Truncating STG_GREATER_BOSTON_CRIME before fresh load")
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_GREATER_BOSTON_CRIME")
        print(f"Loading data from {s3_path} to Snowflake")
        cursor.execute(copy_sql)

        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_GREATER_BOSTON_CRIME")
        count = cursor.fetchone()[0]
        print(f"Total records in STG_GREATER_BOSTON_CRIME: {count}")

        return {"status": "success", "records_loaded": count}
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="greater_boston_crime_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 3, 16),
    schedule_interval='0 2 1 * *',    # Monthly 1st at 2am — aggregate data doesn't change daily
    catchup=False,
    tags=["greater_boston", "crime", "s3", "fbi_cde"],
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