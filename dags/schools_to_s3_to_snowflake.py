from __future__ import annotations

import os
import csv
import zipfile
from datetime import datetime
import snowflake.connector
from dbfread import DBF

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
}

def extract_and_convert_to_csv(**context):
    """Extract schools data from shapefile and convert to CSV."""
    
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    
    # Paths
    zip_path = f"{base_dir}/dags/schools.zip"
    csv_path = f"{base_dir}/schools_pt_{ts}.csv"
    
    # Check if ZIP file exists
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Schools shapefile not found at {zip_path}. "
            f"Please download from https://s3.us-east-1.amazonaws.com/download.massgis.digital.mass.gov/shapefiles/state/schools.zip "
            f"and place it in the airflow-project directory."
        )
    
    print(f"Extracting shapefile from {zip_path}")
    
    # Extract ZIP
    extract_dir = f"{base_dir}/schools_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find the DBF file (search in subdirectories too)
    dbf_file = None
    all_files = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(extract_dir):
        all_files.extend(files)
        for file in files:
            if file.lower().endswith('.dbf'):
                dbf_file = os.path.join(root, file)
                print(f"Found DBF file: {file} in {root}")
                break
        if dbf_file:
            break
    
    print(f"All files extracted: {all_files}")
    
    if not dbf_file:
        raise FileNotFoundError(f"No .dbf file found in extracted files. Files: {all_files}")
    
    print(f"Reading DBF file: {dbf_file}")

    # Read DBF and convert to list of dictionaries
    table = DBF(dbf_file, encoding='latin-1')
    records = list(iter(table))
    
    print(f"Found {len(records)} school records")
    
    if not records:
        raise ValueError("No records found in DBF file")
    
    # Write to CSV
    print(f"Converting to CSV: {csv_path}")
    fieldnames = records[0].keys()

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"CSV saved to {csv_path}")
    print(f"Columns: {list(fieldnames)}")
    
    # Cleanup extracted files
    import shutil
    shutil.rmtree(extract_dir)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='csv_path', value=csv_path)
    context['task_instance'].xcom_push(key='timestamp', value=ts)
    
    return {"csv_path": csv_path, "record_count": len(records)}


def upload_to_s3(**context):
    """Upload the schools CSV file to S3."""
    s3_bucket = Variable.get("schools_s3_bucket")
    s3_prefix = Variable.get("schools_s3_key_prefix", default_var="schools/")
    aws_conn_id = Variable.get("schools_aws_conn_id", default_var="aws_default")
    
    ti = context['task_instance']
    csv_path = ti.xcom_pull(task_ids='extract_and_convert', key='csv_path')
    ts = ti.xcom_pull(task_ids='extract_and_convert', key='timestamp')
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    s3_key = f"{s3_prefix}schools_pt_{ts}.csv"
    print(f"Uploading {csv_path} to s3://{s3_bucket}/{s3_key}")
    
    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(
        filename=csv_path,
        key=s3_key,
        bucket_name=s3_bucket,
        replace=True
    )
    
    print(f"Successfully uploaded to S3: s3://{s3_bucket}/{s3_key}")

    # Push S3 location to XCom
    context['task_instance'].xcom_push(key='s3_key', value=s3_key)
    context['task_instance'].xcom_push(key='s3_bucket', value=s3_bucket)
    
    return s3_key

def create_snowflake_table(**context):
    """Create Snowflake table for Massachusetts Schools data."""
    
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
    DROP TABLE IF EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS;
    """
    
    create_table_sql = """
    CREATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS (
        SCHID VARCHAR(20),
        NAME VARCHAR(200),
        ADDRESS VARCHAR(300),
        TOWN_MAIL VARCHAR(100),
        TOWN VARCHAR(100),
        ZIPCODE VARCHAR(10),
        PHONE VARCHAR(20),
        GRADES VARCHAR(50),
        TYPE VARCHAR(10),
        TYPE_DESC VARCHAR(100),
        TYPE_DESC2 VARCHAR(100),
        MA_ADDR_ID VARCHAR(50),
        SHAPE_WKT VARCHAR(1000),
        POINT_X FLOAT,
        POINT_Y FLOAT,
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """

    try:
        cursor = conn.cursor()
        
        print("Dropping old STG_BOSTON_SCHOOLS table if exists...")
        cursor.execute(drop_table_sql)
        
        print("Creating STG_BOSTON_SCHOOLS table in STAGE schema...")
        cursor.execute(create_table_sql)
        
        print("Schools table created successfully")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()


def load_s3_to_snowflake(**context):
    """Load schools CSV from S3 to Snowflake - full reload annually."""
    
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
        
        # Truncate before loading (full replacement)
        print("Truncating table for full reload...")
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS")
        
        print(f"Loading data from {s3_path} to Snowflake")
        copy_sql = f"""
        COPY INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS
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
            NULL_IF = ('NULL', 'null', '', 'None')
        )
        ON_ERROR = 'CONTINUE';
        """
        cursor.execute(copy_sql)
        
        # Get count
        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_SCHOOLS")
        count = cursor.fetchone()[0]
        print(f"Total schools in table: {count}")
        
        return {"status": "success", "records_loaded": count}
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

with DAG(
    dag_id="schools_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 19),
    schedule_interval='0 0 1 9 *',
    catchup=False,
    tags=["schools", "s3", "shapefile", "snowflake"],
) as dag:

    extract_and_convert_task = PythonOperator(
        task_id="extract_and_convert",
        python_callable=extract_and_convert_to_csv,
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
    extract_and_convert_task >> upload_task >> create_table_task >> load_to_snowflake_task