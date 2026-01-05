from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

# In a real setup, we'd import the actual functions from src.train and src.data.ingest
# For MVP, we'll inline or mock the calls to show the structure
# OR better: use BashOperator to run the scripts we already made.

def check_drift():
    logging.info("Checking for data drift...")
    # Real logic: Query Evidently service or check drift report
    # Return True if drift detected
    return True

default_args = {
    'owner': 'traffic-vision',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'traffic_vision_retrain',
    default_args=default_args,
    description='Retraining pipeline triggered by drift or schedule',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    tags=['mlops', 'vision'],
) as dag:

    ingest_new_data = PythonOperator(
        task_id='ingest_new_data',
        python_callable=lambda: logging.info("Ingesting new data (simulated Active Learning)...")
        # In real life: fetch new labeled data from labeling tool
    )

    train_model = PythonOperator(
        task_id='train_model',
        # In real life: ./venv/bin/python src/train.py
        python_callable=lambda: logging.info("Training model...")
    )

    validate_model = PythonOperator(
        task_id='validate_model',
        python_callable=lambda: logging.info("Validating model against holdout set...")
    )

    register_model = PythonOperator(
        task_id='register_model',
        python_callable=lambda: logging.info("Registering new model version to MLflow...")
    )

    deploy_model = PythonOperator(
        task_id='deploy_model',
        python_callable=lambda: logging.info("Triggering deployment (CD)...")
    )

    ingest_new_data >> train_model >> validate_model >> register_model >> deploy_model
