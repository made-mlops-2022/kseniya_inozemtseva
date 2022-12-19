# ScikitLearn + FastAPI + Docker
Deployment of ML models using Python's Scikit-Learn + FastAPI + Docker
## Based on https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea

# Dataset
## Cleveland Heart Disease 

# Train

`$ python code/train.py`

# Web application

`$ uvicorn main:app`

# Docker

`$ docker build . -t sklearn_fastapi_docker_hw`

`$ docker run -p 8000:8000 sklearn_fastapi_docker_hw`

# Test

When running: 

`$ python tests/test_main.py`

pytest:

`$ pytest -s`

# Docker size minimization
Starting with 1.55Gb

1) --no-cache-dir 

2) FROM python:3.8-slim

Result 752 MB