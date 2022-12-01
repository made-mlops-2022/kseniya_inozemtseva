import os
import click
import logging
import sys
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import pandas as pd
import joblib
import gzip


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _preprocess(X, y):
    pipe = Pipeline([
        ('scaler', ColumnTransformer([('scaler', StandardScaler(), X.columns[:2])], remainder='drop')),

    ])

    return pipe.fit_transform(X, y), y


@click.command("preprocess")
@click.argument("input-dir")
@click.argument("output-dir")
def preprocess(input_dir: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('Downloader')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(input_dir, 'preprocessing.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("here we are")

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    X = data
    logger.debug(f'X {data.shape}')
    logger.debug(f'y {y.shape}')
    X, y = _preprocess(X, y)
    logger.debug(f'X {X.shape}')
    logger.debug(f'y {y.shape}')


    #X.to_csv(os.path.join(output_dir, "data.csv"))
    np.savetxt(os.path.join(output_dir, f"data.csv"), X, delimiter=',')
    np.savetxt(os.path.join(output_dir, "target.csv"), y.to_numpy(), delimiter=',')


@click.command("split")
@click.argument("input-dir")
@click.argument("output-dir")
def split(input_dir: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('Splitter')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'splitting.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("here we are")

    X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    logger.debug(f'X {X.shape}')
    logger.debug(f'y {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # X_train.to_csv(os.path.join(output_dir, "X_train.csv"))
    # y_train.to_csv(os.path.join(output_dir, "y_train.csv"))
    # X_test.to_csv(os.path.join(output_dir, "X_test.csv"))
    # y_test.to_csv(os.path.join(output_dir, "y_test.csv"))

    np.savetxt(os.path.join(output_dir, "X_train.csv"), X_train.to_numpy(), delimiter=',')
    np.savetxt(os.path.join(output_dir, "y_train.csv"), y_train.to_numpy(), delimiter=',')
    np.savetxt(os.path.join(output_dir, "X_test.csv"), X_test.to_numpy(), delimiter=',')
    np.savetxt(os.path.join(output_dir, "y_test.csv"), y_test.to_numpy(), delimiter=',')


@click.command("train")
@click.argument("input-dir")
@click.argument("output-dir")
def train(input_dir: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('TrainLogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'traininging.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("here we are")

    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    estimators = []
    estimators.append(('logistic', LogisticRegression()))
    estimators.append(('cart', DecisionTreeClassifier()))
    estimators.append(('svm', SVC()))

    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, y_train)

    joblib.dump(ensemble, gzip.open(os.path.join(output_dir, 'model_binary.dat.gz'), "wb"))


@click.command("validate")
@click.argument("input-dir")
@click.argument("output-dir")
def validate(input_dir: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('ValLogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, 'validation.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug("here we are")

    model = joblib.load(os.path.join(output_dir, 'model_binary.dat.gz'))

    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv"))

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    with open(os.path.join(output_dir, "score.txt"), 'w') as file:
        file.write(f'f1={score}')


if __name__ == '__main__':
    preprocess()
