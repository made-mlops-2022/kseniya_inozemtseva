import click
import logging

import ml_example.data_utils as data1
import ml_example.features_utils as features_utils
import ml_example.models_utils as models_utils
import ml_example.params as params



def train(config_path: str):
    logger = logging.getLogger(__name__)
    logger.info(f'Starting with {config_path=}')

    config = params.read_config(config_path)
    logger.info(f'Loaded config')
    project_structure = config["ProjectStructure"]
    feature_params = config["FeatureParams"]
    model_params = config["ModelParams"]

    data = data1.read_dataset(project_structure.dataset_path)
    logger.info(f'Data read from {project_structure.dataset_path=}')
    logger.info(f'Data {data.columns}')
    logger.info(f'Data  {data.shape=}')

    logger.info(f'Feature building, options {feature_params}')
    processor = features_utils.get_processor(feature_params)
    logger.info('First step of the pipeline constructed')
    logger.debug(f'{processor=}')

    logger.info(f'Constructing model, options {model_params}')
    model = models_utils.get_model(processor,
                                   model_params=model_params)
    logger.info('Model constructed')
    logger.debug(f'{model=}')

    logger.info(f'Splitting datset, test size {feature_params.test_size}')
    X_train, X_val, y_train, y_val = data1.split_train_val_data(
        data, feature_params
    )

    model.fit(X_train, y_train)

    logger.info(f'Model fitted')

    for metric in model_params.metrics:
        score = models_utils.score(model.predict(X_val), y_val, metric)
        logger.info(f'{metric}: {score}')


@click.command(name="train_pipeline")
@click.argument("config_path")
@click.option("--stream", is_flag=True)
@click.option("--debug", is_flag=True)
def launch_train(config_path: str, stream : bool, debug: bool):
    params.tune_logging(debug=True, stream=True)

    logger = logging.getLogger(__name__)
    logger.info(f'Starting with {config_path=}')

    train(config_path)


if __name__ == '__main__':
    params.tune_logging(debug=True, stream=True)

    train("../configs/cle_hea_diss_simple.json")
