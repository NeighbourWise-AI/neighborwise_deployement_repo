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


def fetch_yelp_restaurants(**context):
    """Fetch restaurant data from Yelp API and save as CSV."""
    api_key = Variable.get("yelp_api_key")
    location = Variable.get("yelp_location", default_var="Boston, MA")
    
    ts = context.get("ts_nodash") or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir = "/opt/airflow"
    json_path = f"{base_dir}/yelp_restaurants_{ts}.json"
    csv_path = f"{base_dir}/yelp_restaurants_{ts}.csv"
    
    base_url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Yelp API limits: 50 results per request, max 1000 total per search
    # We'll search multiple categories and neighborhoods to get comprehensive data
    
    all_restaurants = []
    restaurant_ids = set()  # Track unique businesses
    
    # Define search parameters for comprehensive coverage
    categories = ["restaurants", "food", "bars", "cafes"]
    
    # Boston neighborhoods/areas for broader coverage
    locations = [
        "Boston, MA",
        "Cambridge, MA",
        "Somerville, MA",
        "Brookline, MA",
        "Allston, Boston, MA",
        "Back Bay, Boston, MA",
        "Downtown Boston, MA",
        "North End, Boston, MA",
        "South End, Boston, MA",
        "Jamaica Plain, Boston, MA"
    ]
    
    print(f"Fetching restaurants from Yelp API across {len(locations)} locations and {len(categories)} categories")
    
    for loc in locations:
        for category in categories:
            print(f"\nSearching {category} in {loc}...")
            
            offset = 0
            limit = 50  # Yelp max per request
            
            while offset < 1000:  # Yelp API limit
                params = {
                    "location": loc,
                    "categories": category,
                    "limit": limit,
                    "offset": offset
                }
                
                try:
                    resp = requests.get(base_url, headers=headers, params=params, timeout=30)
                    resp.raise_for_status()
                    
                    data = resp.json()
                    businesses = data.get('businesses', [])
                    
                    if not businesses:
                        print(f"  No more results at offset {offset}")
                        break
                    
                    # Add unique restaurants only
                    new_count = 0
                    for business in businesses:
                        if business['id'] not in restaurant_ids:
                            restaurant_ids.add(business['id'])
                            
                            # Flatten the business data
                            restaurant = {
                                'id': business.get('id'),
                                'name': business.get('name'),
                                'alias': business.get('alias'),
                                'image_url': business.get('image_url'),
                                'is_closed': business.get('is_closed', False),
                                'url': business.get('url'),
                                'review_count': business.get('review_count', 0),
                                'rating': business.get('rating', 0),
                                'price': business.get('price', ''),
                                'phone': business.get('phone', ''),
                                'display_phone': business.get('display_phone', ''),
                                
                                # Categories
                                'categories': json.dumps([c.get('title') for c in business.get('categories', [])]),
                                
                                # Location
                                'address1': business.get('location', {}).get('address1', ''),
                                'address2': business.get('location', {}).get('address2', ''),
                                'address3': business.get('location', {}).get('address3', ''),
                                'city': business.get('location', {}).get('city', ''),
                                'state': business.get('location', {}).get('state', ''),
                                'zip_code': business.get('location', {}).get('zip_code', ''),
                                'country': business.get('location', {}).get('country', ''),
                                'display_address': json.dumps(business.get('location', {}).get('display_address', [])),
                                
                                # Coordinates
                                'latitude': business.get('coordinates', {}).get('latitude', None),
                                'longitude': business.get('coordinates', {}).get('longitude', None),
                                
                                # Transactions
                                'transactions': json.dumps(business.get('transactions', [])),
                                
                                # Distance (if available)
                                'distance': business.get('distance', None)
                            }
                            
                            all_restaurants.append(restaurant)
                            new_count += 1
                    
                    print(f"  Offset {offset}: Found {len(businesses)} results, {new_count} new unique")
                    
                    if len(businesses) < limit:
                        break  # No more results for this search
                    
                    offset += limit
                    time.sleep(0.2)  # Rate limiting
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit
                        print(f"  Rate limited, waiting 5 seconds...")
                        time.sleep(5)
                    else:
                        print(f"  Error: {e}")
                        break
            
            time.sleep(1)  # Delay between category searches
    
    print(f"\nFinished fetching. Total unique restaurants: {len(all_restaurants)}")
    
    if not all_restaurants:
        raise ValueError("No restaurants found")
    
    # Save JSON
    print(f"Saving JSON to {json_path}")
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump({"restaurants": all_restaurants}, fh)
    
    # Save CSV
    print(f"Converting {len(all_restaurants)} restaurants to CSV")
    fieldnames = all_restaurants[0].keys()
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_restaurants)
    
    print(f"CSV saved to {csv_path}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='csv_path', value=csv_path)
    context['task_instance'].xcom_push(key='json_path', value=json_path)
    context['task_instance'].xcom_push(key='timestamp', value=ts)
    
    return {"csv_path": csv_path, "json_path": json_path, "restaurant_count": len(all_restaurants)}


def upload_to_s3(**context):
    """Upload the restaurants CSV file to S3."""
    s3_bucket = Variable.get("yelp_s3_bucket")
    s3_prefix = Variable.get("yelp_s3_key_prefix", default_var="restaurants/")
    aws_conn_id = Variable.get("yelp_aws_conn_id", default_var="aws_default")
    
    ti = context['task_instance']
    csv_path = ti.xcom_pull(task_ids='fetch_restaurants', key='csv_path')
    ts = ti.xcom_pull(task_ids='fetch_restaurants', key='timestamp')
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    s3_key = f"{s3_prefix}yelp_restaurants_{ts}.csv"
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
    """Create Snowflake table for Yelp restaurant data."""
    
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
    DROP TABLE IF EXISTS NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_RESTAURANTS;
    """
    
    create_table_sql = """
    CREATE OR REPLACE TABLE NEIGHBOURWISE_DOMAINS.STAGE.STG_BOSTON_RESTAURANTS (
        ID VARCHAR(100),
        NAME VARCHAR(200),
        ALIAS VARCHAR(200),
        IMAGE_URL VARCHAR(500),
        IS_CLOSED BOOLEAN,
        URL VARCHAR(500),
        REVIEW_COUNT NUMBER,
        RATING FLOAT,
        PRICE VARCHAR(10),
        PHONE VARCHAR(50),
        DISPLAY_PHONE VARCHAR(50),
        CATEGORIES VARCHAR(500),
        ADDRESS1 VARCHAR(200),
        ADDRESS2 VARCHAR(200),
        ADDRESS3 VARCHAR(200),
        CITY VARCHAR(100),
        STATE VARCHAR(50),
        ZIP_CODE VARCHAR(20),
        COUNTRY VARCHAR(50),
        DISPLAY_ADDRESS VARCHAR(500),
        LATITUDE FLOAT,
        LONGITUDE FLOAT,
        TRANSACTIONS VARCHAR(200),
        DISTANCE FLOAT,
        LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    );
    """
    
    try:
        cursor = conn.cursor()
        
        print("Dropping old STG_BOSTON_RESTAURANTS table if exists...")
        cursor.execute(drop_table_sql)
        
        print("Creating STG_BOSTON_RESTAURANTS table in STAGE schema...")
        cursor.execute(create_table_sql)
        
        print("Restaurants table created successfully")
        return "Table ready"
    finally:
        cursor.close()
        conn.close()


def load_s3_to_snowflake(**context):
    """Load restaurants CSV from S3 to Snowflake with TRUNCATE and INSERT."""
    
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
            CREATE OR REPLACE TEMPORARY TABLE RESTAURANTS_TEMP 
            LIKE STG_BOSTON_RESTAURANTS;
        """)
        
        # Load CSV into temp table
        print(f"Loading data from {s3_path} into temp table")
        copy_sql = f"""
        COPY INTO RESTAURANTS_TEMP
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
        ON_ERROR = 'CONTINUE';
        """
        cursor.execute(copy_sql)
        
        # Check temp table count
        cursor.execute("SELECT COUNT(*) FROM RESTAURANTS_TEMP")
        temp_count = cursor.fetchone()[0]
        print(f"Records loaded into temp table: {temp_count}")
        
        # Truncate and insert
        print("Truncating main table for full reload...")
        cursor.execute("TRUNCATE TABLE STG_BOSTON_RESTAURANTS")
        
        print("Inserting restaurant records into main table...")
        cursor.execute("INSERT INTO STG_BOSTON_RESTAURANTS SELECT * FROM RESTAURANTS_TEMP")
        
        # Get final count
        cursor.execute("SELECT COUNT(*) FROM STG_BOSTON_RESTAURANTS")
        count = cursor.fetchone()[0]
        print(f"Total restaurants in table: {count}")
        
        return {"status": "success", "total_records": count}
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


with DAG(
    dag_id="yelp_restaurants_to_s3_to_snowflake",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 2, 19),
    schedule_interval='0 0 1 1,4,7,10 *',  # Quarterly: Jan 1, Apr 1, Jul 1, Oct 1
    catchup=False,
    tags=["yelp", "restaurants", "s3", "api", "snowflake"],
) as dag:

    fetch_restaurants_task = PythonOperator(
        task_id="fetch_restaurants",
        python_callable=fetch_yelp_restaurants,
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
    fetch_restaurants_task >> upload_task >> create_table_task >> load_to_snowflake_task