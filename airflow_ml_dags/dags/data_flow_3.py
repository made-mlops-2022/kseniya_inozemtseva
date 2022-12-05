import os
import sys
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _start():
    sys.stdout.write('Get data. Starting..')


with DAG(
        "data_flow3",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
        catchup=True
) as dag:
    start = PythonOperator(
        task_id="starting", python_callable=_start,
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="/data/raw/{{ ds }} /data/predictions/{{ ds }} \"{{ var.value.MODELPATH }}\"",
        network_mode="bridge",
        task_id="predicions",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=r"F:\study\MADE\MADE22\mlops\hw3\airflow_ml_dags\data", target="/data", type='bind')]
    )

    start >> predict
