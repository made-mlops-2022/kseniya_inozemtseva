import pandas as pd
from ms import model
import numpy as np
from sklearn.exceptions import NotFittedError


def model_ready():
    try:
        model.predict(
            pd.DataFrame({
            "age": 56,
            "sex": 1,
            "cp": 0,
            "trestbps": 160,
            "chol": 234,
            "fbs": 1,
            "restecg": 2,
            "thalach": 131,
            "exang": 0,
            "oldpeak": 0.1,
            "slope": 1,
            "ca": 1,
            "thal": 0
        }, index=[0])
    )
    except NotFittedError:
        return False
    else:
        return True


def predict(X, model):
    prediction = model.predict(X)
    return prediction


def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    print(X)
    predictions = predict(X, model)
    print(predictions, type(predictions), predictions.shape)
    response = {
        'prediction': predictions.tolist(),
        'label': np.apply_along_axis(lambda x: 'disease' if x else 'no disease', 0, predictions[:, None]).tolist()
    }
    print(response)
    return response


