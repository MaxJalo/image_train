from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from docker.types import Mount
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_idempotency(batch_id, **kwargs):
    results_path = f'/opt/airflow/results/{batch_id}.json'
    if os.path.exists(results_path):
        print(f"Batch {batch_id} already processed. Skipping.")
        raise AirflowSkipException(f"Batch {batch_id} already processed")
    return True

dag = DAG(
    'batch_service_dag',
    default_args=default_args,
    description='Батчевый сервис для обработки вагонов',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

check_idempotency_task = PythonOperator(
    task_id='check_idempotency',
    python_callable=check_idempotency,
    op_kwargs={'batch_id': '{{ ds_nodash }}'},
    dag=dag,
)

process_batch_task = DockerOperator(
    task_id='process_batch',
    image='wagon-batch-processor:latest',
    command='--batch-id {{ ds_nodash }}',
    auto_remove=True,
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    mounts=[
        Mount(source='C:/Users/Asus/image_train/airflow/results', target='/app/results', type='bind'),
        Mount(source='C:/Users/Asus/image_train/airflow/backup', target='/opt/airflow/backup', type='bind'),
    ],
    dag=dag,
)

backup_task = BashOperator(
    task_id='backup',
    bash_command='cp /app/results/{{ ds_nodash }}.json /opt/airflow/backup/results_{{ ds_nodash }}.json',
    dag=dag,
)

check_idempotency_task >> process_batch_task >> backup_task