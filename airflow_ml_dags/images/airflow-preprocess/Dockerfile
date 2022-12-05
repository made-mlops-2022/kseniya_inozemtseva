FROM airflow-ml-base:latest

COPY preprocess.py /preprocess.py
COPY setup.py /setup.py

RUN python -m pip install --editable .

#ENTRYPOINT ["python", "preprocess.py"]
ENTRYPOINT []
