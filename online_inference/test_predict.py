import json

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_good():

    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"age": 56, "sex": 1, "cp": 0, "trestbps": 160,
                         "chol": 234, "fbs": 1, "restecg": 2, "thalach": 131,
                         "exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1, "thal": 0 }))
    print(response)
    assert response.status_code == 200
    assert response.json() =={
        "label": ["no disease"],
        "prediction": [0]
    }


def test_bad():

    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"age": 200, "sex": 1, "cp": 0, "trestbps": 160,
                         "chol": 234, "fbs": 1, "restecg": 2, "thalach": 131,
                         "exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1, "thal": 0 }))
    assert response.status_code == 400
    d_resp = json.loads(response.json())

    assert d_resp == [{'ctx': {'limit_value': 130},
             'loc': ['body', 'age'],
             'msg': 'ensure this value is less than 130',
             'type': 'value_error.number.not_lt'}]


def test_validation():


    response = client.post(
        "/predict",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"age": 0, "sex": 1, "cp": 0, "trestbps": 160,
                         "chol": 234, "fbs": 1, "restecg": 2, "thalach": 131,
                         "exang": 0, "oldpeak": 0.1, "slope": 1, "ca": 1, "thal": 0 }))
    assert response.status_code == 400

