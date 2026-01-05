from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from docker.types import Mount

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'retraining_pipeline',
    default_args=default_args,
    description='A simple retraining pipeline triggered by drift',
    schedule_interval=None, # Triggered externally by Evidently script or API
    start_date=days_ago(1),
    tags=['mlops', 'traffic-vision'],
) as dag:

    # Task 1: Check Data (Placeholder - could be DVC pull)
    validate_data = BashOperator(
        task_id='validate_data_integrity',
        bash_command='echo "Data integrity check passed (Simulated)"'
    )

    # Task 2: Train Model using Docker
    # We use the same image we built for the API, but run the training script
    train_model = DockerOperator(
        task_id='train_yolov8',
        image='traffic-vision-api:latest',
        api_version='auto',
        auto_remove=True,
        command='python src/train.py',
        docker_url='unix://var/run/docker.sock',
        mount_tmp_dir=False, # Fix for Mac Docker bind issue
        network_mode='traffic-vision_default', # Connect to same network if needed
        mounts=[
            Mount(source='/Users/abhinavmaurya/Projects/Embitel/traffic-vision/data', target='/app/data', type='bind'),
            Mount(source='/Users/abhinavmaurya/Projects/Embitel/traffic-vision/models', target='/app/models', type='bind'),
            Mount(source='/Users/abhinavmaurya/Projects/Embitel/traffic-vision/mlruns', target='/app/mlruns', type='bind'),
        ],
        # environment={
        #    'MLFLOW_TRACKING_URI': 'http://host.docker.internal:5000' 
        # }
    )

    # Task 3: Notify/Register (Placeholder)
    notify_success = BashOperator(
        task_id='notify_completion',
        bash_command='echo "Model trained and registered successfully!"'
    )

    # Task 4: Hot Swap Model in Production
    # Triggers the /reload endpoint on the API container
    hot_swap_model = BashOperator(
        task_id='hot_swap_model',
        bash_command='curl -X POST http://api:8000/reload || echo "Failed to reload model, API might be down"'
    )

    validate_data >> train_model >> notify_success >> hot_swap_model
