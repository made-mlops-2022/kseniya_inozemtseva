from typing import List, Any
import os
import urllib.request
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from ml_example.params import ProjectStructure


def download_dataset(
        dataset_url,
        data_dir=''
) -> None:

    if not data_dir:
        data_dir = ProjectStructure.data_raw_path

    archive_file = os.path.join(data_dir, "dataset_archive.zip")
    if not os.path.exists(archive_file):
        urllib.request.urlretrieve(dataset_url, archive_file)

    with zipfile.ZipFile(os.path.normpath(archive_file), 'r') as zip_ref:
        dataset_filename = zip_ref.namelist()[0]
        zip_ref.extractall(data_dir)

        dataset_path = os.path.join(data_dir, dataset_filename)

    return dataset_path


def read_dataset(dataset_path='') -> pd.DataFrame:
    if dataset_path:
        return pd.read_csv(os.path.abspath(dataset_path))


def process_data(df : pd.DataFrame) -> None:
    pass


def split_train_val_data(data : pd.DataFrame, feature_params) -> Any:
    X_train, X_val, y_train, y_val = train_test_split(data.drop([feature_params.target_col], axis=1),
                                                      data[feature_params.target_col],
                                                      test_size=feature_params.test_size,
                                                      random_state=42)
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    download_dataset("https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/download?datasetVersionNumber=1")
