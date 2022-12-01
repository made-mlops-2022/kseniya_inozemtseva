import os
from pathlib import Path
import pandas as pd
import joblib
import click
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

@click.command("predict")
@click.argument("input-dir")
@click.argument("output-dir")
@click.argument("model-path")
def predict(input_dir: str, output_dir: str, model_path: str):
    os.makedirs(output_dir, exist_ok=True)

    print(model_path)
    print(Path(model_path).parent)
    print(Path(model_path).parent.exists())
    model = joblib.load(model_path)
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    pipe = Pipeline([
        ('scaler', ColumnTransformer([('scaler', StandardScaler(), data.columns[:2])], remainder='drop')),

    ])

    data = pipe.fit_transform(data)

    data = np.hstack((data, model.predict(data)[:, None]))

    np.savetxt(os.path.join(output_dir, f"data.csv"), data, delimiter=',')


if __name__ == '__main__':
    predict()
