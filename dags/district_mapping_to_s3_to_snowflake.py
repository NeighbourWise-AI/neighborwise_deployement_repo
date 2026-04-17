from __future__ import annotations

import os
import csv
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import snowflake.connector

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
}

BPD_URL = "https://police.boston.gov/districts/"

def scrape_and_convert_to_csv(**context):
    """Scrape BPD districts page and save as CSV locally."""
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    base_dir = "/opt/airflow"
    csv_path = f"{base_dir}/district_mapping_{ts}.csv"

    print(f"Scraping BPD districts from {BPD_URL}")
    response = requests.get(BPD_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    districts = []
    for heading in soup.find_all("h4"):
        text = heading.get_text(strip=True)
        if "(" in text and ")" in text:
            code_part = text.split("(")[0].strip()
            name_part = text.split("(")[1].replace(")", "").strip().upper()
            codes = [c.strip() for c in code_part.split("&")]
            names = [n.strip() for n in name_part.split("&")]

            # If codes and names match in count, map them 1:1
            # e.g. "A1 & A15 (Downtown & Charlestown)" -> A1:Downtown, A15:Charlestown
            # If counts don't match, assign full combined name to all codes
            for i, code in enumerate(codes):
                if code:
                    name = names[i] if i < len(names) else name_part
                    districts.append({
                        "DISTRICT_CODE": code.strip(),
                        "DISTRICT_NAME": name.strip()
                    })

    # Diagnostic print — shows exactly what was scraped before dedup
    print("=== RAW SCRAPED DATA ===")
    print(f"Total raw records before dedup: {len(districts)}")
    for d in districts:
        print(f"  {d['DISTRICT_CODE']} -> {d['DISTRICT_NAME']}")

    if not districts:
        raise ValueError("No districts scraped — check BPD website structure")

    # Deduplicate on DISTRICT_CODE
    seen = set()
    unique_districts = []
    for d in districts:
        if d["DISTRICT_CODE"] not in seen:
            seen.add(d["DISTRICT_CODE"])
            unique_districts.append(d)
    districts = unique_districts

    # Diagnostic print — shows what remains after dedup
    print("=== DEDUPLICATED DATA ===")
    print(f"Total unique records after dedup: {len(districts)}")
    for d in districts:
        print(f"  {d['DISTRICT_CODE']} -> {d['DISTRICT_NAME']}")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["DISTRICT_CODE", "DISTRICT_NAME"])
        writer.writeheader()
        writer.writerows(districts)

    print(f"CSV saved to {csv_path} with {len(districts)} records")

    context["task_instance"].xcom_push(key="csv_path", value=csv_path)
    context["task_instance"].xcom_push(key="timestamp", value=ts)

    return {"csv_path": csv_path, "record_count": len(districts)}


def upload_to_s3(**context):
    """Upload the district mapping CSV to S3 crime-safety folder."""
    s3_bucket   = Variable.get("boston_s3_bucket")
    s3_prefix   = Variable.get("boston_s3_key_prefix", default_var="crime-safety/")
    aws_conn_id = Variable.get("boston_aws_conn_id", default_var="aws_default")

    ti       = context["task_instance"]
    csv_path = ti.xcom_pull(task_ids="scrape_and_convert", key="csv_path")
    ts       = ti.xcom_pull(task_ids="scrape_and_convert", key="timestamp")

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    s3_key = f"{s3_prefix}district_mapping_{ts}.csv"
    print(f"Uploading {csv_path} to s3://{s3_bucket}/{s3_key}")

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(
        filename=csv_path,
        key=s3_key,
        bucket_name=s3_bucket,
        replace=True
    )

    print(f"Successfully uploaded to S3: s3://{s3_bucket}/{s3_key}")

    context["task_instance"].xcom_push(key="s3_key", value=s3_key)
    context["task_instance"].xcom_push(key="s3_bucket", value=s3_bucket)

    return s3_key


def create_snowflake_table(**context):
    """Create STG_DISTRICT_MAPPING table in Snowflake STAGE schema."""
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True
    )

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_DISTRICT_MAPPING (
        DISTRICT_CODE   VARCHAR(10),
        DISTRICT_NAME   VARCHAR(200),
        LOAD_TIMESTAMP  TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """

    try:
        cursor = conn.cursor()
        print("Creating Snowflake table STG_DISTRICT_MAPPING")
        cursor.execute(create_table_sql)
        print("Table STG_DISTRICT_MAPPING created or already exists")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()


def load_s3_to_snowflake(**context):
    """Truncate and reload STG_DISTRICT_MAPPING from S3."""
    ti        = context["task_instance"]
    s3_bucket = ti.xcom_pull(task_ids="upload_to_s3", key="s3_bucket")
    s3_key    = ti.xcom_pull(task_ids="upload_to_s3", key="s3_key")

    from airflow.hooks.base import BaseHook
    aws_conn       = BaseHook.get_connection("aws_default")
    aws_access_key = aws_conn.login
    aws_secret_key = aws_conn.password

    s3_path = f"s3://{s3_bucket}/{s3_key}"

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"],
        insecure_mode=True
    )

    try:
        cursor = conn.cursor()
        cursor.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")

        print("Truncating STG_DISTRICT_MAPPING")
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_DISTRICT_MAPPING")

        copy_sql = f"""
        COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_DISTRICT_MAPPING
            (DISTRICT_CODE, DISTRICT_NAME)
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

        print(f"Loading data from {s3_path} to Snowflake")
        cursor.execute(copy_sql)

        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_DISTRICT_MAPPING")
        count = cursor.fetchone()[0]
        print(f"Total records in STG_DISTRICT_MAPPING: {count}")

        return {"status": "success", "records_loaded": count}

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="district_mapping_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 25),
    schedule_interval="@yearly",
    catchup=False,
    tags=["boston", "crime", "reference", "districts"],
) as dag:

    scrape_and_convert_task = PythonOperator(
        task_id="scrape_and_convert",
        python_callable=scrape_and_convert_to_csv,
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

    scrape_and_convert_task >> upload_task >> create_table_task >> load_to_snowflake_task