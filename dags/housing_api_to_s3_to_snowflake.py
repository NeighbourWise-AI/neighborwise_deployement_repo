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
    """Fetch housing/property assessment data from API with pagination, save locally, and convert to CSV."""
    api_url = Variable.get("housing_api_url")
    
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    json_path = f"{base_dir}/housing_{ts}.json"
    csv_path = f"{base_dir}/housing_{ts}.csv"
    
    print(f"Fetching ALL housing data from {api_url} with pagination")
    
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
    
    print(f"Finished fetching. Total records: {len(all_records)}")
    
    if not all_records:
        raise ValueError("No records found in API response")
    
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
    s3_bucket = Variable.get("housing_s3_bucket")
    s3_prefix = Variable.get("housing_s3_key_prefix", default_var="housing/")
    aws_conn_id = Variable.get("housing_aws_conn_id", default_var="aws_default")
    
    ti = context['task_instance']
    csv_path = ti.xcom_pull(task_ids='fetch_and_convert', key='csv_path')
    ts = ti.xcom_pull(task_ids='fetch_and_convert', key='timestamp')
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    s3_key = f"{s3_prefix}housing_{ts}.csv"
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
    """Create Snowflake table for housing data with correct schema."""
    
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
    DROP TABLE IF EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING;
    """
    
    create_table_sql = """
        CREATE OR REPLACE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING (
        _ID VARCHAR(50),
        PID VARCHAR(50),
        CM_ID VARCHAR(50),
        GIS_ID VARCHAR(50),
        ST_NUM VARCHAR(20),
        ST_NUM2 VARCHAR(20),
        ST_NAME VARCHAR(200),
        UNIT_NUM VARCHAR(50),
        CITY VARCHAR(100),
        ZIP_CODE VARCHAR(10),
        BLDG_SEQ VARCHAR(50),
        NUM_BLDGS VARCHAR(50),
        LUC VARCHAR(10),
        LU VARCHAR(10),
        LU_DESC VARCHAR(200),
        BLDG_TYPE VARCHAR(50),
        OWN_OCC VARCHAR(10),
        OWNER VARCHAR(300),
        MAIL_ADDRESSEE VARCHAR(200),
        MAIL_STREET_ADDRESS VARCHAR(300),
        MAIL_CITY VARCHAR(100),
        MAIL_STATE VARCHAR(50),
        MAIL_ZIP_CODE VARCHAR(50),
        RES_FLOOR VARCHAR(50),
        CD_FLOOR VARCHAR(50),
        RES_UNITS VARCHAR(50),
        COM_UNITS VARCHAR(50),
        RC_UNITS VARCHAR(50),
        LAND_SF VARCHAR(50),
        GROSS_AREA VARCHAR(50),
        LIVING_AREA VARCHAR(50),
        LAND_VALUE VARCHAR(50),
        BLDG_VALUE VARCHAR(50),
        SFYI_VALUE VARCHAR(50),
        TOTAL_VALUE VARCHAR(50),
        GROSS_TAX VARCHAR(50),
        YR_BUILT VARCHAR(50),
        YR_REMODEL VARCHAR(50),
        STRUCTURE_CLASS VARCHAR(50),
        ROOF_STRUCTURE VARCHAR(50),
        ROOF_COVER VARCHAR(50),
        INT_WALL VARCHAR(50),
        EXT_FNISHED VARCHAR(50),
        INT_COND VARCHAR(50),
        EXT_COND VARCHAR(50),
        OVERALL_COND VARCHAR(50),
        BED_RMS VARCHAR(50),
        FULL_BTH VARCHAR(50),
        HLF_BTH VARCHAR(50),
        KITCHENS VARCHAR(50),
        TT_RMS VARCHAR(50),
        BDRM_COND VARCHAR(50),
        BTHRM_STYLE1 VARCHAR(50),
        BTHRM_STYLE2 VARCHAR(50),
        BTHRM_STYLE3 VARCHAR(50),
        KITCHEN_TYPE VARCHAR(50),
        KITCHEN_STYLE1 VARCHAR(50),
        KITCHEN_STYLE2 VARCHAR(50),
        KITCHEN_STYLE3 VARCHAR(50),
        HEAT_TYPE VARCHAR(50),
        HEAT_SYSTEM VARCHAR(50),
        AC_TYPE VARCHAR(50),
        FIREPLACES VARCHAR(50),
        ORIENTATION VARCHAR(50),
        NUM_PARKING VARCHAR(50),
        PROP_VIEW VARCHAR(50),
        CORNER_UNIT VARCHAR(50)
    );
    """
    
    try:
        cursor = conn.cursor()
        
        print("Dropping old STG_BOSTON_HOUSING table if exists...")
        cursor.execute(drop_table_sql)
        
        print("Creating STG_BOSTON_HOUSING table in STAGE schema...")
        cursor.execute(create_table_sql)
        
        print("Housing table created successfully")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()


def load_s3_to_snowflake(**context):
    """Load housing CSV from S3 to Snowflake using MERGE to avoid duplicates."""
    
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

        print("Setting schema to STAGE...")
        cursor.execute("USE SCHEMA NEIGHBOURWISE_DOMAINS.STAGE")
        
        # Create temporary staging table
        print("Creating temporary staging table...")
        cursor.execute("""
            CREATE OR REPLACE TEMPORARY TABLE HOUSING_TEMP 
            LIKE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING;
        """)
        
        # Load CSV into temp table (excluding LOAD_TIMESTAMP which is auto-generated)
        # Load CSV into temp table
        print(f"Loading data from {s3_path} into temp table")
        copy_sql = f"""
        COPY INTO HOUSING_TEMP
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
        )
        ON_ERROR = 'ABORT_STATEMENT';
        """
        cursor.execute(copy_sql)

        # Check how many loaded into temp table
        cursor.execute("SELECT COUNT(*) FROM HOUSING_TEMP")
        temp_count = cursor.fetchone()[0]
        print(f"Records loaded into HOUSING_TEMP: {temp_count}")

        # Also check a sample
        cursor.execute("SELECT * FROM HOUSING_TEMP LIMIT 5")
        sample = cursor.fetchall()
        print(f"Sample from HOUSING_TEMP: {sample}")
        
        # Truncate and insert all records (full reload)
        print("Truncating main table for full reload...")
        cursor.execute("TRUNCATE TABLE STG_BOSTON_HOUSING")

        print("Inserting housing records into main table...")
        insert_sql = """
        INSERT INTO STG_BOSTON_HOUSING
        SELECT * FROM HOUSING_TEMP;
        """
        cursor.execute(insert_sql)
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_HOUSING")
        count = cursor.fetchone()[0]
        print(f"Total housing records in table: {count}")
        
        return {"status": "success", "total_records": count}
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

with DAG(
    dag_id="housing_api_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 19),
    schedule_interval='0 0 1 7 *',  # Annually on July 1st (fiscal year)
    catchup=False,
    tags=["housing", "property", "assessment", "s3", "api", "snowflake"],
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