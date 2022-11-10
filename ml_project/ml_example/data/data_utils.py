from pathlib import Path
from typing import List, Any
import os
import urllib.request
import zipfile
import io

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_example.params.params_utils import ProjectStructure


def read_dataset(dataset_path='') -> pd.DataFrame:
    if dataset_path:
        return pd.read_csv(dataset_path)


def split_train_val_data(data : pd.DataFrame, feature_params) -> Any:
    X_train, X_val, y_train, y_val = train_test_split(data.drop([feature_params.target_col], axis=1),
                                                      data[feature_params.target_col],
                                                      test_size=feature_params.test_size,
                                                      random_state=42)
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':

    os.chdir(Path(__file__).parent.parent)
    project_structure = ProjectStructure()
    Path(project_structure.dataset_path).parent.mkdir(parents=True, exist_ok=True)
    # prepare_dataset(
    #     "https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci/download?datasetVersionNumber=1",
    #     project_structure.dataset_path
    # )
