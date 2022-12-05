import os
import sys
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _start():
    sys.stdout.write('Get data. Starting..')


with DAG(
        "data_flow2",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(5),
        catchup=True
) as dag:
    start = PythonOperator(
        task_id="starting", python_callable=_start,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="preprocess /data/raw/{{ ds }} /data/preprocessed/{{ ds }}",
        network_mode="bridge",
        task_id="data_preprocessing",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=r"F:\study\MADE\MADE22\mlops\hw3\airflow_ml_dags\data", target="/data", type='bind')]
    )
    split = DockerOperator(
        image="airflow-preprocess",
        command="split /data/preprocessed/{{ ds }} /data/preprocessed/{{ ds }}",
        network_mode="bridge",
        task_id="data_splitting",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=r"F:\study\MADE\MADE22\mlops\hw3\airflow_ml_dags\data", target="/data", type='bind')]
    )

    train = DockerOperator(
        image="airflow-preprocess",
        command="train /data/preprocessed/{{ ds }} /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="data_training",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=r"F:\study\MADE\MADE22\mlops\hw3\airflow_ml_dags\data", target="/data", type='bind')]
    )

    validate = DockerOperator(
        image="airflow-preprocess",
        command="validate /data/preprocessed/{{ ds }} /data/models/{{ ds }}",
        network_mode="bridge",
        task_id="data_validation",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=r"F:\study\MADE\MADE22\mlops\hw3\airflow_ml_dags\data", target="/data", type='bind')]
    )

    finish = BashOperator(
        task_id="finishing",
        bash_command=f"echo Finished.",
    )

    start >> preprocess >> split >> train >> validate >> finish
