from __future__ import annotations

import os
import json
import csv
from datetime import datetime
import requests
import time
import snowflake.connector

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 1,
}

def fetch_mbta_routes(**context):
    """Fetch MBTA routes data and save locally as CSV."""
    api_key = Variable.get("mbta_api_key")
    base_url = "https://api-v3.mbta.com"

    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    
    routes_csv = f"{base_dir}/mbta_routes_{ts}.csv"

    print("Fetching MBTA routes...")
    url = f"{base_url}/routes?api_key={api_key}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    data = response.json()
    routes_list = []
    
    route_types = {
        0: 'Light Rail',
        1: 'Heavy Rail (Subway)',
        2: 'Commuter Rail',
        3: 'Bus',
        4: 'Ferry'
    }

    for route in data['data']:
        routes_list.append({
            'route_id': route['id'],
            'route_type': route['attributes']['type'],
            'route_type_name': route_types.get(route['attributes']['type'], 'Unknown'),
            'long_name': route['attributes'].get('long_name', ''),
            'short_name': route['attributes'].get('short_name', ''),
            'description': route['attributes'].get('description', ''),
            'color': route['attributes'].get('color', ''),
            'fare_class': route['attributes'].get('fare_class', ''),
            # ADD THESE TWO LINES:
            'direction_names': json.dumps(route['attributes'].get('direction_names', [])),
            'direction_destinations': json.dumps(route['attributes'].get('direction_destinations', []))
        })
    
    print(f"Found {len(routes_list)} routes")

    # Write to CSV
    if routes_list:
        fieldnames = routes_list[0].keys()
        with open(routes_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(routes_list)
        
        print(f"Routes CSV saved to {routes_csv}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='routes_csv', value=routes_csv)
    context['task_instance'].xcom_push(key='timestamp', value=ts)

    return {"routes_csv": routes_csv, "route_count": len(routes_list)}

def fetch_mbta_stops(**context):
    """Fetch MBTA stops data and save locally as CSV."""
    api_key = Variable.get("mbta_api_key")
    base_url = "https://api-v3.mbta.com"
    
    ti = context['task_instance']
    ts = ti.xcom_pull(task_ids='fetch_routes', key='timestamp')
    base_dir = "/opt/airflow"
    
    stops_csv = f"{base_dir}/mbta_stops_{ts}.csv"
    
    print("Fetching MBTA stops...")
    url = f"{base_url}/stops?api_key={api_key}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    stops_list = []

    for stop in data['data']:
        stops_list.append({
            'stop_id': stop['id'],
            'stop_name': stop['attributes']['name'],
            'description': stop['attributes'].get('description', ''),
            'latitude': stop['attributes']['latitude'],
            'longitude': stop['attributes']['longitude'],
            'wheelchair_boarding': stop['attributes'].get('wheelchair_boarding', 0),
            'municipality': stop['attributes'].get('municipality', ''),
            'platform_code': stop['attributes'].get('platform_code', ''),
            'platform_name': stop['attributes'].get('platform_name', ''),
            # ADD THESE THREE LINES:
            'address': stop['attributes'].get('address', ''),
            'at_street': stop['attributes'].get('at_street', ''),
            'vehicle_type': stop['attributes'].get('vehicle_type', '')
        })
    
    print(f"Found {len(stops_list)} stops")

    # Write to CSV
    if stops_list:
        fieldnames = stops_list[0].keys()
        with open(stops_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stops_list)
        
        print(f"Stops CSV saved to {stops_csv}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='stops_csv', value=stops_csv)
    return {"stops_csv": stops_csv, "stop_count": len(stops_list)}

def fetch_mbta_route_stop_mapping(**context):
    """Fetch route-stop relationships WITH direction and sequence."""
    api_key = Variable.get("mbta_api_key")
    base_url = "https://api-v3.mbta.com"
    
    ti = context['task_instance']
    ts = ti.xcom_pull(task_ids='fetch_routes', key='timestamp')
    
    base_dir = "/opt/airflow"
    mapping_csv = f"{base_dir}/mbta_route_stops_mapping_{ts}.csv"
    
    print("Fetching route-stop relationships with directions...")
    
    # Re-fetch routes to get route IDs and names
    url = f"{base_url}/routes?api_key={api_key}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    routes_data = response.json()['data']
    
    # Create route lookup for names and types
    route_types = {0: 'Light Rail', 1: 'Heavy Rail (Subway)', 2: 'Commuter Rail', 3: 'Bus', 4: 'Ferry'}
    route_lookup = {
        r['id']: {
            'name': r['attributes'].get('long_name', ''),
            'type': route_types.get(r['attributes']['type'], 'Unknown')
        }
        for r in routes_data
    }
    
    # Re-fetch stops to get municipality and coordinates
    url = f"{base_url}/stops?api_key={api_key}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    stops_data = response.json()['data']
    
    # Create stop lookup
    stop_lookup = {
        s['id']: {
            'municipality': s['attributes'].get('municipality', ''),
            'latitude': s['attributes'].get('latitude', ''),
            'longitude': s['attributes'].get('longitude', ''),
            'wheelchair_boarding': s['attributes'].get('wheelchair_boarding', 0)
        }
        for s in stops_data
    }
    
    route_stop_mapping = []
    total_routes = len(routes_data)
    
    for idx, route in enumerate(routes_data, 1):
        route_id = route['id']
        route_info = route_lookup[route_id]
        
        print(f"Processing {idx}/{total_routes}: {route_id} ({route_info['name']})...")
        
        # Fetch stops for BOTH directions (0 and 1)
        for direction in [0, 1]:
            url = f"{base_url}/stops?filter[route]={route_id}&filter[direction_id]={direction}&api_key={api_key}"
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                stops_in_direction = response.json()['data']
                
                for seq, stop in enumerate(stops_in_direction, 1):
                    stop_id = stop['id']
                    stop_info = stop_lookup.get(stop_id, {})
                    
                    route_stop_mapping.append({
                        'route_id': route_id,
                        'route_name': route_info['name'],
                        'route_type': route_info['type'],
                        'stop_id': stop_id,
                        'stop_name': stop['attributes']['name'],
                        'direction_id': direction,
                        'direction_name': 'Outbound' if direction == 0 else 'Inbound',
                        'stop_sequence': seq,
                        'municipality': stop_info.get('municipality', ''),
                        'latitude': stop_info.get('latitude', ''),
                        'longitude': stop_info.get('longitude', ''),
                        'wheelchair_accessible': stop_info.get('wheelchair_boarding', 0)
                    })
            
            time.sleep(0.2)
        
        # Rate limiting
        if idx % 10 == 0:
            time.sleep(2)
    
    print(f"\nFound {len(route_stop_mapping)} route-stop connections")
    
    # Write to CSV
    if route_stop_mapping:
        fieldnames = route_stop_mapping[0].keys()
        with open(mapping_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(route_stop_mapping)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='mapping_csv', value=mapping_csv)
    
    return {"mapping_csv": mapping_csv, "mapping_count": len(route_stop_mapping)}

def upload_to_s3(**context):
    """Upload all three CSV files to S3."""
    s3_bucket = Variable.get("mbta_s3_bucket")
    s3_prefix = Variable.get("mbta_s3_key_prefix", default_var="mbta-data/")
    aws_conn_id = Variable.get("mbta_aws_conn_id", default_var="aws_default")
    
    ti = context['task_instance']
    ts = ti.xcom_pull(task_ids='fetch_routes', key='timestamp')
    routes_csv = ti.xcom_pull(task_ids='fetch_routes', key='routes_csv')
    stops_csv = ti.xcom_pull(task_ids='fetch_stops', key='stops_csv')
    mapping_csv = ti.xcom_pull(task_ids='fetch_mapping', key='mapping_csv')

    hook = S3Hook(aws_conn_id=aws_conn_id)
    uploaded_keys = []
    
    # Upload routes
    if routes_csv and os.path.exists(routes_csv):
        s3_key = f"{s3_prefix}mbta_routes_{ts}.csv"
        print(f"Uploading routes to s3://{s3_bucket}/{s3_key}")
        hook.load_file(filename=routes_csv, key=s3_key, bucket_name=s3_bucket, replace=True)
        uploaded_keys.append(s3_key)
    
    # Upload stops
    if stops_csv and os.path.exists(stops_csv):
        s3_key = f"{s3_prefix}mbta_stops_{ts}.csv"
        print(f"Uploading stops to s3://{s3_bucket}/{s3_key}")
        hook.load_file(filename=stops_csv, key=s3_key, bucket_name=s3_bucket, replace=True)
        uploaded_keys.append(s3_key)
    
    # Upload mapping
    if mapping_csv and os.path.exists(mapping_csv):
        s3_key = f"{s3_prefix}mbta_route_stops_mapping_{ts}.csv"
        print(f"Uploading mapping to s3://{s3_bucket}/{s3_key}")
        hook.load_file(filename=mapping_csv, key=s3_key, bucket_name=s3_bucket, replace=True)
        uploaded_keys.append(s3_key)
    
    print(f"Successfully uploaded {len(uploaded_keys)} files to S3")

    # Push S3 info to XCom
    context['task_instance'].xcom_push(key='s3_bucket', value=s3_bucket)
    context['task_instance'].xcom_push(key='uploaded_keys', value=uploaded_keys)
    context['task_instance'].xcom_push(key='s3_prefix', value=s3_prefix)
    
    return {"uploaded_files": len(uploaded_keys), "s3_keys": uploaded_keys}

def create_snowflake_tables(**context):
    """Create three Snowflake tables for MBTA data with enhanced route planning fields."""
    
    conn = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        role=os.environ['SNOWFLAKE_ROLE'],
        insecure_mode=True
    )
    
    create_routes_table = """
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_ROUTES (
        ROUTE_ID VARCHAR(50) PRIMARY KEY,
        ROUTE_TYPE NUMBER,
        ROUTE_TYPE_NAME VARCHAR(50),
        LONG_NAME VARCHAR(200),
        SHORT_NAME VARCHAR(50),
        DESCRIPTION VARCHAR(500),
        COLOR VARCHAR(10),
        FARE_CLASS VARCHAR(50),
        DIRECTION_NAMES VARCHAR(500),
        DIRECTION_DESTINATIONS VARCHAR(500),
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """
    
    create_stops_table = """
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_STOPS (
        STOP_ID VARCHAR(50) PRIMARY KEY,
        STOP_NAME VARCHAR(200),
        DESCRIPTION VARCHAR(500),
        LATITUDE FLOAT,
        LONGITUDE FLOAT,
        WHEELCHAIR_BOARDING NUMBER,
        MUNICIPALITY VARCHAR(100),
        PLATFORM_CODE VARCHAR(50),
        PLATFORM_NAME VARCHAR(100),
        ADDRESS VARCHAR(300),
        AT_STREET VARCHAR(200),
        VEHICLE_TYPE VARCHAR(50),
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """
    
    create_mapping_table = """
    CREATE TABLE IF NOT EXISTS NEIGHBOURWISE_DOMAINS.INTERMEDIATE.INT_BOSTON_MBTA_MAPPING (
        ROUTE_ID VARCHAR(50),
        ROUTE_NAME VARCHAR(200),
        ROUTE_TYPE VARCHAR(50),
        STOP_ID VARCHAR(50),
        STOP_NAME VARCHAR(200),
        DIRECTION_ID NUMBER,
        DIRECTION_NAME VARCHAR(20),
        STOP_SEQUENCE NUMBER,
        MUNICIPALITY VARCHAR(100),
        LATITUDE FLOAT,
        LONGITUDE FLOAT,
        WHEELCHAIR_ACCESSIBLE NUMBER,
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        PRIMARY KEY (ROUTE_ID, STOP_ID, DIRECTION_ID)
    );
    """
    
    try:
        cursor = conn.cursor()
        
        print("Creating STG_BOSTON_MBTA_ROUTES table in STAGE schema...")
        cursor.execute(create_routes_table)
        
        print("Creating STG_BOSTON_MBTA_STOPS table in STAGE schema...")
        cursor.execute(create_stops_table)
        
        print("Creating INT_BOSTON_MBTA_MAPPING table in INTERMEDIATE schema...")
        cursor.execute(create_mapping_table)
        
        print("All MBTA tables created successfully")
        return "Tables ready"
    finally:
        cursor.close()
        conn.close()

def load_s3_to_snowflake(**context):
    """Load all three CSV files from S3 to Snowflake using MERGE/TRUNCATE strategy."""
    
    ti = context['task_instance']
    s3_bucket = ti.xcom_pull(task_ids='upload_to_s3', key='s3_bucket')
    s3_prefix = ti.xcom_pull(task_ids='upload_to_s3', key='s3_prefix')
    ts = ti.xcom_pull(task_ids='fetch_routes', key='timestamp')
    
    from airflow.hooks.base import BaseHook
    aws_conn = BaseHook.get_connection('aws_default')
    aws_access_key = aws_conn.login
    aws_secret_key = aws_conn.password
    
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
        
        # ===== ROUTES - MERGE (routes can be added/updated) =====
        routes_s3_path = f"s3://{s3_bucket}/{s3_prefix}mbta_routes_{ts}.csv"
        print(f"Loading routes from {routes_s3_path}")
        
        cursor.execute("CREATE OR REPLACE TEMPORARY TABLE MBTA_ROUTES_TEMP LIKE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_ROUTES;")
        
        copy_routes_sql = f"""
        COPY INTO MBTA_ROUTES_TEMP
        (ROUTE_ID, ROUTE_TYPE, ROUTE_TYPE_NAME, LONG_NAME, SHORT_NAME, 
         DESCRIPTION, COLOR, FARE_CLASS, DIRECTION_NAMES, DIRECTION_DESTINATIONS)
        FROM '{routes_s3_path}'
        CREDENTIALS = (AWS_KEY_ID = '{aws_access_key}' AWS_SECRET_KEY = '{aws_secret_key}')
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' TRIM_SPACE = TRUE NULL_IF = ('NULL', 'null', '', 'None'))
        ON_ERROR = 'CONTINUE';
        """
        cursor.execute(copy_routes_sql)
        
        print("Merging routes...")
        cursor.execute("""
        MERGE INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_ROUTES AS target
        USING MBTA_ROUTES_TEMP AS source
        ON target.ROUTE_ID = source.ROUTE_ID
        WHEN MATCHED THEN
            UPDATE SET
                ROUTE_TYPE = source.ROUTE_TYPE,
                ROUTE_TYPE_NAME = source.ROUTE_TYPE_NAME,
                LONG_NAME = source.LONG_NAME,
                SHORT_NAME = source.SHORT_NAME,
                DESCRIPTION = source.DESCRIPTION,
                COLOR = source.COLOR,
                FARE_CLASS = source.FARE_CLASS,
                DIRECTION_NAMES = source.DIRECTION_NAMES,
                DIRECTION_DESTINATIONS = source.DIRECTION_DESTINATIONS
        WHEN NOT MATCHED THEN
            INSERT (ROUTE_ID, ROUTE_TYPE, ROUTE_TYPE_NAME, LONG_NAME, SHORT_NAME, 
                    DESCRIPTION, COLOR, FARE_CLASS, DIRECTION_NAMES, DIRECTION_DESTINATIONS)
            VALUES (source.ROUTE_ID, source.ROUTE_TYPE, source.ROUTE_TYPE_NAME, source.LONG_NAME, 
                    source.SHORT_NAME, source.DESCRIPTION, source.COLOR, source.FARE_CLASS, 
                    source.DIRECTION_NAMES, source.DIRECTION_DESTINATIONS);
        """)
        
        # ===== STOPS - MERGE (stops can be added/updated) =====
        stops_s3_path = f"s3://{s3_bucket}/{s3_prefix}mbta_stops_{ts}.csv"
        print(f"Loading stops from {stops_s3_path}")
        
        cursor.execute("CREATE OR REPLACE TEMPORARY TABLE MBTA_STOPS_TEMP LIKE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_STOPS;")
        
        copy_stops_sql = f"""
        COPY INTO MBTA_STOPS_TEMP
        (STOP_ID, STOP_NAME, DESCRIPTION, LATITUDE, LONGITUDE, 
         WHEELCHAIR_BOARDING, MUNICIPALITY, PLATFORM_CODE, PLATFORM_NAME,
         ADDRESS, AT_STREET, VEHICLE_TYPE)
        FROM '{stops_s3_path}'
        CREDENTIALS = (AWS_KEY_ID = '{aws_access_key}' AWS_SECRET_KEY = '{aws_secret_key}')
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' TRIM_SPACE = TRUE NULL_IF = ('NULL', 'null', '', 'None'))
        ON_ERROR = 'CONTINUE';
        """
        cursor.execute(copy_stops_sql)
        
        print("Merging stops...")
        cursor.execute("""
        MERGE INTO NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_STOPS AS target
        USING MBTA_STOPS_TEMP AS source
        ON target.STOP_ID = source.STOP_ID
        WHEN MATCHED THEN
            UPDATE SET
                STOP_NAME = source.STOP_NAME,
                DESCRIPTION = source.DESCRIPTION,
                LATITUDE = source.LATITUDE,
                LONGITUDE = source.LONGITUDE,
                WHEELCHAIR_BOARDING = source.WHEELCHAIR_BOARDING,
                MUNICIPALITY = source.MUNICIPALITY,
                PLATFORM_CODE = source.PLATFORM_CODE,
                PLATFORM_NAME = source.PLATFORM_NAME,
                ADDRESS = source.ADDRESS,
                AT_STREET = source.AT_STREET,
                VEHICLE_TYPE = source.VEHICLE_TYPE
        WHEN NOT MATCHED THEN
            INSERT (STOP_ID, STOP_NAME, DESCRIPTION, LATITUDE, LONGITUDE, 
                    WHEELCHAIR_BOARDING, MUNICIPALITY, PLATFORM_CODE, PLATFORM_NAME,
                    ADDRESS, AT_STREET, VEHICLE_TYPE)
            VALUES (source.STOP_ID, source.STOP_NAME, source.DESCRIPTION, source.LATITUDE, 
                    source.LONGITUDE, source.WHEELCHAIR_BOARDING, source.MUNICIPALITY, 
                    source.PLATFORM_CODE, source.PLATFORM_NAME, source.ADDRESS, 
                    source.AT_STREET, source.VEHICLE_TYPE);
        """)
        
        # ===== MAPPING - DELETE OLD + INSERT NEW (full snapshot approach) =====
        # For mapping, we want to replace ALL data since it's a complete snapshot of relationships
        mapping_s3_path = f"s3://{s3_bucket}/{s3_prefix}mbta_route_stops_mapping_{ts}.csv"
        print(f"Loading mapping from {mapping_s3_path}")
        
        # Delete all old mapping data and insert fresh
        print("Truncating mapping table for full reload...")
        cursor.execute("TRUNCATE TABLE NEIGHBOURWISE_DOMAINS.INTERMEDIATE.INT_BOSTON_MBTA_MAPPING")
        
        copy_mapping_sql = f"""
        COPY INTO NEIGHBOURWISE_DOMAINS.INTERMEDIATE.INT_BOSTON_MBTA_MAPPING
        (ROUTE_ID, ROUTE_NAME, ROUTE_TYPE, STOP_ID, STOP_NAME, 
         DIRECTION_ID, DIRECTION_NAME, STOP_SEQUENCE, MUNICIPALITY, 
         LATITUDE, LONGITUDE, WHEELCHAIR_ACCESSIBLE)
        FROM '{mapping_s3_path}'
        CREDENTIALS = (AWS_KEY_ID = '{aws_access_key}' AWS_SECRET_KEY = '{aws_secret_key}')
        FILE_FORMAT = (TYPE = 'CSV' SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '"' TRIM_SPACE = TRUE NULL_IF = ('NULL', 'null', '', 'None'))
        ON_ERROR = 'CONTINUE';
        """
        cursor.execute(copy_mapping_sql)
        
        # Get final counts
        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_ROUTES")
        routes_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_MBTA_STOPS")
        stops_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM NEIGHBOURWISE_DOMAINS.INTERMEDIATE.INT_BOSTON_MBTA_MAPPING")
        mapping_count = cursor.fetchone()[0]
        
        print(f"\nData loaded/merged successfully!")
        print(f"ROUTES: {routes_count} total records")
        print(f"STOPS: {stops_count} total records")
        print(f"MAPPING: {mapping_count} total records (full snapshot)")
        
        return {
            "status": "success",
            "routes_loaded": routes_count,
            "stops_loaded": stops_count,
            "mappings_loaded": mapping_count
        }
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()
    
with DAG(
    dag_id="mbta_api_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 19),
    schedule_interval='0 0 1 1 *',
    catchup=False,
    tags=["mbta", "s3", "api", "snowflake", "transit"],
) as dag:

    fetch_routes_task = PythonOperator(
        task_id="fetch_routes",
        python_callable=fetch_mbta_routes,
        provide_context=True,
    )
    
    fetch_stops_task = PythonOperator(
        task_id="fetch_stops",
        python_callable=fetch_mbta_stops,
        provide_context=True,
    )
    
    fetch_mapping_task = PythonOperator(
        task_id="fetch_mapping",
        python_callable=fetch_mbta_route_stop_mapping,
        provide_context=True,
    )
    
    upload_task = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3,
        provide_context=True,
    )
    
    create_tables_task = PythonOperator(
        task_id="create_snowflake_tables",
        python_callable=create_snowflake_tables,
        provide_context=True,
    )
    
    load_to_snowflake_task = PythonOperator(
        task_id="load_to_snowflake",
        python_callable=load_s3_to_snowflake,
        provide_context=True,
    )

    # Task dependencies: Fetch all 3 datasets in parallel, then upload, then create tables, then load
    [fetch_routes_task, fetch_stops_task, fetch_mapping_task] >> upload_task >> create_tables_task >> load_to_snowflake_task