import os
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass, field, asdict, make_dataclass
import json
import logging
import logging.config


@dataclass()
class ModelParams:
    type: str
    metrics: Any
    params_tunning: bool = False
    params: Optional[Any] = None
    params_grid: Optional[Any] = None
    random_state: int = 42

    def __post_init__(self):
        if self.type == 'RandomForest':
            if self.params_tunning:
                if self.params_grid:
                    self.params_grid = RandomForestParamsGrid(**self.params_grid)
                else:
                    logger = logging.getLogger(self.__class__.__name__)
                    logger.error('Invalid configuration')
                    raise ValueError
            elif self.params:
                self.params = RandomForestParams(**self.params)
            else:
                logger = logging.getLogger(self.__class__.__name__)
                logger.error('Invalid configuration')
                raise ValueError


@dataclass()
class RandomForestParams:
    max_depth: int = 10
    max_features: str = 'sqrt'
    min_samples_leaf: int = 1
    min_samples_split: int = 5
    n_estimators: int = 100


@dataclass()
class RandomForestParamsGrid:
    randomforestclassifier__bootstrap: List = field(default_factory=lambda: [True, False])
    randomforestclassifier__max_depth: List = field(default_factory=lambda: [10, 20, 30, None]) #, 40, 50, 60, 70, 80, 90, 100, None])
    randomforestclassifier__max_features: List = field(default_factory=lambda: ['auto', 'sqrt'])
    randomforestclassifier__min_samples_leaf: List = field(default_factory=lambda: [1, 2, 4])
    randomforestclassifier__min_samples_split: List = field(default_factory=lambda: [2, 5, 10])
    randomforestclassifier__n_estimators: List = field(default_factory=lambda: [200, 300, 400]) #, 800, 1000, 1200, 1400, 1600, 1800, 2000])


@dataclass()
class ProjectStructure:
    dataset_path: str = "../data/raw/heart_cleveland_upload.csv"
    data_raw_path: str = "../data/raw"
    data_proc_path: str = "../data/processed"

    models_path: str = "../models"


@dataclass()
class FeatureParams:
    test_size: float = 0.2
    target_col: str = 'condition'
    categorical_cols: List[str] = field(default_factory=lambda: ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    numerical_cols: List[str] = field(default_factory=lambda: ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
    cols_to_imput: List[str] = field(default_factory=lambda: {'num': [], 'cat' : []})
    cols_to_ohe: List[str] = field(default_factory=lambda: [])
    scale_params: List[str] = field(default_factory=lambda: (['oldpeak','trestbps'], [0.4, 10]))
    cols_to_std_scale: List[str] = field(default_factory=lambda: ['age'])
    cat_impute_strat: str = 'most_frequent'
    num_impute_strat: str = 'constant'


class_names = {ModelParams.__name__: ModelParams,
               RandomForestParams.__name__: RandomForestParams,
               RandomForestParamsGrid.__name__: RandomForestParamsGrid,
               ProjectStructure.__name__: ProjectStructure,
               FeatureParams.__name__: FeatureParams}


def save_config():
    config = dict()
    config[ModelParams.__name__] = asdict(ModelParams(
        type='RandomForest',
        metrics=['f1'],
        params_tunning=True,
        params_grid=asdict(RandomForestParamsGrid()),
        params=None
    ))

    config[ProjectStructure.__name__] = asdict(ProjectStructure(
        dataset_path="../data/raw/heart_cleveland_upload.csv",
        data_raw_path="../data/raw",
        data_proc_path="../data/processed",
        models_path="../models"
    ))

    config[FeatureParams.__name__] = asdict(FeatureParams(
        test_size=0.2,
        target_col='condition',
        categorical_cols=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
        numerical_cols=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
        cols_to_imput={'num': [], 'cat': []},
        cols_to_ohe=[],
        scale_params=(['oldpeak', 'trestbps'], [0.4, 10]),
        cols_to_std_scale=['age'],
        cat_impute_strat='most_frequent',
        num_impute_strat='constant'
    ))

    savepath = os.path.join("D:\study\MADE\MADE22\mlops\ml_project\configs", 'cle_hea_diss.json')

    with open(savepath, 'w') as fileobj:
        json.dump(config, fileobj, indent=4)

    config[ModelParams.__name__] = asdict(ModelParams(
        type='RandomForest',
        metrics=['f1'],
        params_tunning=False,
        params=asdict(RandomForestParams()),
        params_grid=None
    ))

    savepath = os.path.join("D:\study\MADE\MADE22\mlops\ml_project\configs", 'cle_hea_diss_simple.json')

    with open(savepath, 'w') as fileobj:
        json.dump(config, fileobj, indent=4)


def read_config(config_path: str):
    logger = logging.getLogger(__name__)
    try:
        with open(config_path) as fileobj:
            d = json.load(fileobj)
    except Exception as e:
        logger.debug(e, exc_info=True)
        raise e

    params = dict()
    try:
        for key, val in d.items():
            params[key] = class_names[key](**val)
    except ValueError as e:
        logger.debug(d)
        raise e
    except Exception as e:
        logger.debug(e, exc_info=True)
        raise e
    return params


log_conf = {
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - [%(levelname)s] -  %(name)s - "
                      "(%(filename)s).%(funcName)s(%(lineno)d) - %(message)s",
        },
        "short": {
            "format": "[%(levelname)s] -  %(name)10s - %(message)s",
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": "ml_project.log",
            "formatter": "detailed",

        },
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["file_handler"]
        },
    }
}
log_stream_handlr_conf = {
    "level": "INFO",
    "formatter": "short",
    "class": "logging.StreamHandler",
    "stream": "ext://sys.stdout",
}


def tune_logging(debug, stream):
    if debug:
        log_conf["loggers"][""]["level"] = "DEBUG"
    if stream:
        console_log = "stream_handler"
        log_conf["handlers"][console_log] = log_stream_handlr_conf
        log_conf["loggers"][""]["handlers"].append(console_log)
    logging.config.dictConfig(log_conf)
    logging.info("New launch")


if __name__ == '__main__':
    tune_logging(debug=True, stream=True)
    save_config()
    read_config("D:\study\MADE\MADE22\mlops\ml_project\configs/cle_hea_diss.json")
