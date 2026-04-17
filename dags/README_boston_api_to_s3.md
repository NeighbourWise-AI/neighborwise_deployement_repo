# Boston API → S3 DAG

This DAG fetches data from a configured Boston data API URL and uploads the response to S3.

Setup
1. Add Airflow Variables (via UI or CLI):

```bash
airflow variables set boston_api_crime_url "https://data.boston.gov/api/1/...&resource_id=ee73430d-96c0-423e-ad21-c4cfb5..."
airflow variables set boston_s3_bucket "my-bucket-name"
# optional:
airflow variables set boston_s3_key_prefix "boston/"
airflow variables set boston_aws_conn_id "aws_default"
```

2. Ensure an Airflow AWS connection exists named in `boston_aws_conn_id` (default `aws_default`). You can add it via the UI or the CLI:

```bash
airflow connections add aws_default \
  --conn-type aws \
  --conn-extra '{"aws_access_key_id":"YOUR_KEY","aws_secret_access_key":"YOUR_SECRET"}'
```

Run
- Trigger the DAG `boston_api_to_s3` from the UI or CLI.

Notes
- The DAG writes a temporary file to `/tmp` before uploading and removes it after upload.
- Adjust `start_date` / `schedule_interval` in `dags/boston_api_to_s3.py` as needed.
