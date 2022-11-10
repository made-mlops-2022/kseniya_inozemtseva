import os
import click
import logging

import pandas as pd

import ml_project.data as data_utils
import ml_project.features as features_utils
import ml_project.models as models_utils
import ml_project.params as params
from ml_project.tests import synthetic_data


def train(config_path: str):
    logger = logging.getLogger(__name__)
    logger.info(f'Starting training with {config_path=}')

    config = params.read_config(config_path)
    logger.info(f'Loaded config')
    project_structure = config["ProjectStructure"]
    feature_params = config["FeatureParams"]
    model_params = config["ModelParams"]

    data = pd.read_csv(project_structure.dataset_path)
    logger.info(f'Data read from {project_structure.dataset_path=}')
    logger.info(f'Data {data.columns}')
    logger.info(f'Data  {data.shape=}')

    logger.info(f'Feature building, options {feature_params}')
    preprocessor = features_utils.get_preprocessor(feature_params)
    logger.info('First step of the pipeline constructed')

    logger.info(f'Constructing model, options {model_params}')
    model = models_utils.get_model(preprocessor,
                                   model_params=model_params)
    logger.info('Model constructed')

    logger.info(f'Splitting datset, test size {feature_params.test_size}')
    X_train, X_val, y_train, y_val = data_utils.split_train_val_data(
        data, feature_params
    )

    model.fit(X_train, y_train)

    logger.info(f'Model fitted')

    for metric in model_params.metrics:
        score = models_utils.score(model.predict(X_val), y_val, metric)
        logger.info(f'{metric}: {score}')

    logger.info('Saving state')

    models_utils.save_state(model, project_structure, model_params, feature_params)


# TODO: option logger_path
@click.command(name="train_pipeline")
@click.argument("config_path",
                type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--stream", is_flag=True)
@click.option("--debug", is_flag=True)
def launch_train(config_path: str, stream: bool, debug: bool):
    params.tune_logging(debug, stream)

    logger = logging.getLogger(__name__)
    logger.info(f'Starting with {config_path=}')

    train(config_path)


def run_train_from_here():
    params.tune_logging(debug=True, stream=True)
    os.chdir('..')
    train("./configs/cle_hea_diss_simple.json")


def predict(model, data):
    logger = logging.getLogger(__name__)
    y_pred = model.predict(data)
    logger.info(f'Predictions are ready.')

    return y_pred


@click.command(name="predict")
@click.argument("data_path",
                type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("model_path",
                type=click.Path(exists=False, file_okay=True, dir_okay=False)
                )
@click.option("--output_path",
              type=click.Path(exists=False, file_okay=True, dir_okay=False),
              prompt="where to save predictions?",
              default='output.txt',
              help='output file')
@click.option("--stream", is_flag=True)
@click.option("--debug", is_flag=True)
def load_predict(data_path, model_path, output_path, stream=True, debug=True):
    params.tune_logging(debug, stream)

    logger = logging.getLogger(__name__)
    logger.info(f'Starting prediction session for {data_path=}')
    logger.info(f'Model {model_path=}')
    model = models_utils.restore_model(model_path)

    data = data_utils.read_dataset(data_path)
    logger.info(f'Data read from {data_path=}')
    logger.info(f'Data {data.columns}')
    logger.info(f'Data  {data.shape=}')
    y_pred = predict(model, data)
    pd.DataFrame(y_pred).to_csv(output_path, index=False)
    logger.info(f'Predictions written {output_path=}')


if __name__ == '__main__':
    os.chdir(params.get_project_root())
    config_path = "../configs/cle_hea_diss_simple.json"
    data_path = "../test.csv"
    test_data = synthetic_data.create_data_like("../data/raw/heart_cleveland_upload.csv", 'condition', 100)
    test_data.drop(columns=['condition']).to_csv(data_path, index=False)
    model_path = "../models/RandomForest.pkl"

    params.tune_logging(debug=True, stream=True)

    logger = logging.getLogger(__name__)
    logger.info(f'Starting prediction session with {config_path=}')
    if config_path:
        config = params.read_config(config_path)
        logger.info(f'Loaded config')
        project_structure = config["ProjectStructure"]
        feature_params = config["FeatureParams"]
        model_params = config["ModelParams"]
        model = models_utils.restore_model(model_path)
    else:
        model, project_structure, \
        model_params, feat_params = models_utils.models_utils.restore_state(model_path)

    data = data_utils.read_dataset(data_path)
    logger.info(f'Data read from {data_path=}')
    logger.info(f'Data {data.columns}')
    logger.info(f'Data  {data.shape=}')

    predict(model, data)
