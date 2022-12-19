import requests
import pandas as pd

if __name__ == '__main__':
    dataset = pd.read_csv('..\data\heart_cleveland_upload.csv')
    dataset = dataset[dataset.condition == 1]
    dataset.pop('condition')
    for _, row in dataset.head(100).iterrows():
        response = requests.post(
            'http://localhost:8000/predict',
            headers={"Content-Type": "application/json"},
            data=row.to_json()
        )
        print(response.json())