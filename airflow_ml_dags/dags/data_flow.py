import os
import sys
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _start():
    sys.stdout.write('Get data. Starting..')


with DAG(
        "data_flow",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
        catchup=True
) as dag:
    start = PythonOperator(
        task_id="starting", python_callable=_start,
    )
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="data_flow_task",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=r"F:\study\MADE\MADE22\mlops\hw3\airflow_ml_dags\data", target="/data", type='bind')]
    )

    finish = BashOperator(
        task_id="finishing",
        bash_command=f"echo Finished.",
    )

    start >> download >> finish
