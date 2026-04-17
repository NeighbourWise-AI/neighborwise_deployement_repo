from __future__ import annotations

import os
import json
import csv
from datetime import datetime
import requests
import snowflake.connector

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
}

def fetch_and_convert_to_csv(**context):
    """Fetch Blue Bikes station data from API with pagination, save locally, and convert to CSV."""
    api_url = Variable.get("bluebikes_api_url")
    
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    json_path = f"{base_dir}/bluebikes_stations_{ts}.json"
    csv_path = f"{base_dir}/bluebikes_stations_{ts}.csv"

    print(f"Fetching ALL Blue Bikes station data from {api_url} with pagination")
    
    all_records = []
    offset = 0
    limit = 10000
    
    while True:
        paginated_url = f"{api_url}&limit={limit}&offset={offset}"
        print(f"Fetching records {offset} to {offset + limit}...")
        
        resp = requests.get(paginated_url, timeout=120)
        resp.raise_for_status()
        
        data = resp.json()
        records = data.get('result', {}).get('records', [])
        
        if not records:
            print("No more records found")
            break
        
        all_records.extend(records)
        print(f"Fetched {len(records)} records. Total so far: {len(all_records)}")
        
        if len(records) < limit:
            print("Reached last page")
            break
        
        offset += limit
    
    # Save complete JSON
    print(f"Saving JSON to {json_path}")
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump({"records": all_records}, fh)
    
    # Convert to CSV
    print(f"Converting {len(all_records)} records to CSV")
    fieldnames = all_records[0].keys()
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    
    print(f"CSV saved to {csv_path}")
    print(f"JSON and CSV files saved in airflow-project directory")
    
    # Push file paths to XCom for the next task
    context['task_instance'].xcom_push(key='csv_path', value=csv_path)
    context['task_instance'].xcom_push(key='json_path', value=json_path)
    context['task_instance'].xcom_push(key='timestamp', value=ts)
    
    return {"csv_path": csv_path, "json_path": json_path, "record_count": len(all_records)}

def upload_to_s3(**context):
    """Upload the CSV file to S3."""
    s3_bucket = Variable.get("bluebikes_s3_bucket")
    s3_prefix = Variable.get("bluebikes_s3_key_prefix", default_var="bluebikes/")
    aws_conn_id = Variable.get("bluebikes_aws_conn_id", default_var="aws_default")
    
    ti = context['task_instance']
    csv_path = ti.xcom_pull(task_ids='fetch_and_convert', key='csv_path')
    ts = ti.xcom_pull(task_ids='fetch_and_convert', key='timestamp')
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    s3_key = f"{s3_prefix}bluebikes_stations_{ts}.csv"
    print(f"Uploading {csv_path} to s3://{s3_bucket}/{s3_key}")

    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(
        filename=csv_path,
        key=s3_key,
        bucket_name=s3_bucket,
        replace=True
    )
    
    print(f"Successfully uploaded to S3: s3://{s3_bucket}/{s3_key}")
    
    # Push S3 location to XCom for Snowflake task
    context['task_instance'].xcom_push(key='s3_key', value=s3_key)
    context['task_instance'].xcom_push(key='s3_bucket', value=s3_bucket)
    
    return s3_key

def create_snowflake_table(**context):
    """Create Snowflake table for Blue Bikes station data."""
    
    conn = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        role=os.environ['SNOWFLAKE_ROLE'],
        insecure_mode=True
    )
    
    drop_table_sql = """
    DROP TABLE IF EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_BLUEBIKES_STATIONS;
    """
    
    create_table_sql = """
    CREATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_BLUEBIKES_STATIONS (
        _ID NUMBER,
        NUMBER VARCHAR(50),
        NAME VARCHAR(200),
        LATITUDE FLOAT,
        LONGITUDE FLOAT,
        DISTRICT VARCHAR(50),
        PUBLIC_ VARCHAR(10),
        TOTAL_DOCKS NUMBER,
        SHAPE_WKT VARCHAR(1000),
        POINT_X FLOAT,
        POINT_Y FLOAT,
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """
    
    try:
        cursor = conn.cursor()
        
        print("Dropping old STG_BOSTON_BLUEBIKES_STATIONS table if exists...")
        cursor.execute(drop_table_sql)
        
        print("Creating STG_BOSTON_BLUEBIKES_STATIONS table in STAGE schema...")
        cursor.execute(create_table_sql)
        
        print("Blue Bikes table created successfully")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()

def load_s3_to_snowflake(**context):
    """Load Blue Bikes CSV from S3 to Snowflake using MERGE to avoid duplicates."""
    
    ti = context['task_instance']
    s3_bucket = ti.xcom_pull(task_ids='upload_to_s3', key='s3_bucket')
    s3_key = ti.xcom_pull(task_ids='upload_to_s3', key='s3_key')
    
    # Get AWS credentials
    from airflow.hooks.base import BaseHook
    aws_conn = BaseHook.get_connection('aws_default')
    aws_access_key = aws_conn.login
    aws_secret_key = aws_conn.password
    
    s3_path = f"s3://{s3_bucket}/{s3_key}"
    
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        role=os.environ['SNOWFLAKE_ROLE'],
        insecure_mode=True
    )
    
    try:
        cursor = conn.cursor()
        
        # Create temporary staging table
        print("Creating temporary staging table...")
        cursor.execute("""
            CREATE OR REPLACE TEMPORARY TABLE BLUEBIKES_TEMP 
            LIKE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_BLUEBIKES_STATIONS;
        """)
        
        # Load CSV into temp table
        print(f"Loading data from {s3_path} into temp table")
        copy_sql = f"""
        COPY INTO BLUEBIKES_TEMP
        (_ID, NUMBER, NAME, LATITUDE, LONGITUDE, DISTRICT, PUBLIC_, TOTAL_DOCKS, SHAPE_WKT, POINT_X, POINT_Y)
        FROM '{s3_path}'
        CREDENTIALS = (
            AWS_KEY_ID = '{aws_access_key}'
            AWS_SECRET_KEY = '{aws_secret_key}'
        )
        FILE_FORMAT = (
            TYPE = 'CSV'
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            TRIM_SPACE = TRUE
            ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
            NULL_IF = ('NULL', 'null', '', 'None')
        )
        ON_ERROR = 'CONTINUE';
        """
        cursor.execute(copy_sql)
        
        # MERGE from temp to main table (use NUMBER as unique key)
        print("Merging new/updated stations into main table...")
        merge_sql = """
        MERGE INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_BLUEBIKES_STATIONS AS target
        USING BLUEBIKES_TEMP AS source
        ON target.NUMBER = source.NUMBER
        WHEN MATCHED THEN
            UPDATE SET
                _ID = source._ID,
                NAME = source.NAME,
                LATITUDE = source.LATITUDE,
                LONGITUDE = source.LONGITUDE,
                DISTRICT = source.DISTRICT,
                PUBLIC_ = source.PUBLIC_,
                TOTAL_DOCKS = source.TOTAL_DOCKS,
                SHAPE_WKT = source.SHAPE_WKT,
                POINT_X = source.POINT_X,
                POINT_Y = source.POINT_Y
        WHEN NOT MATCHED THEN
            INSERT (_ID, NUMBER, NAME, LATITUDE, LONGITUDE, DISTRICT, PUBLIC_, 
                    TOTAL_DOCKS, SHAPE_WKT, POINT_X, POINT_Y)
            VALUES (source._ID, source.NUMBER, source.NAME, source.LATITUDE, source.LONGITUDE,
                    source.DISTRICT, source.PUBLIC_, source.TOTAL_DOCKS, source.SHAPE_WKT,
                    source.POINT_X, source.POINT_Y);
        """
        cursor.execute(merge_sql)
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_BLUEBIKES_STATIONS")
        count = cursor.fetchone()[0]
        print(f"Total Blue Bikes stations in table: {count}")
        
        return {"status": "success", "records_loaded": count}
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()
    
with DAG(
    dag_id="bluebikes_api_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 19),
    schedule_interval='0 0 1 */3 *',
    catchup=False,
    tags=["bluebikes", "s3", "api", "snowflake", "transit"],
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
    
    # Define task dependencies
    fetch_and_convert_task >> upload_task >> create_table_task >> load_to_snowflake_task