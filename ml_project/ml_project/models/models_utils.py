import os
import pickle
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import get_scorer
from dataclasses import asdict


def get_model(
        pipel, model_params
):
    if model_params.type == 'RandomForest':
        if model_params.params_tunning:
            grid = GridSearchCV(make_pipeline(pipel, RandomForestClassifier(random_state=model_params.random_state)),
                                param_grid=asdict(model_params.params_grid),
                                cv=2,
                                scoring=model_params.metrics[0])

            return grid
        else:
            return make_pipeline(pipel, RandomForestClassifier(random_state=model_params.random_state,
                                          **asdict(model_params.params)))
    else:
        raise NotImplementedError


def score(predictions, true_values, score_func):
    return get_scorer(score_func)._score_func(
        true_values,
        predictions
    )


def save_state(pipeline, project_structure, model_params, feat_params):
    logger = logging.getLogger(__name__)

    savefile = os.path.join(project_structure.models_path, model_params.type)

    idx = 0
    existing = []
    while os.path.exists(savefile + '.pkl'):
        existing.append(savefile + '.pkl')
        idx += 1
        savefile = savefile.split('_')[0] + f'_{idx}'

    logger.info(f'saving model and parameters to {savefile}')

    savefile = savefile + '.pkl'

    obj_to_save = [pipeline, project_structure, model_params, feat_params]
    with open(savefile, 'wb') as fileobj:
        pickle.dump(obj_to_save, fileobj)
    print(f'Saved to {savefile}')


def restore_state(filepath, restore_config=True):
    logger = logging.getLogger(__name__)

    with open(filepath, 'rb') as fileobj:
        obj1 = pickle.load(fileobj, encoding="latin1")

    if not len(obj1) == 4:
        logger.error('Corrupted dump')
        raise TypeError

    predict = getattr(obj1[0], "predict", None)
    if not predict or not callable(predict):
        logger.error('Corrupted dump. First part is not a model.')
        raise TypeError
    if restore_config:
        dataset_path = getattr(obj1[1], "dataset_path", None)
        if not dataset_path:
            logger.error('Corrupted dump. Second part is not a project structure.')
            raise TypeError

        model_type = getattr(obj1[2], "type", None)
        if not model_type:
            logger.error('Corrupted dump. No model parameters in the third part.')
            raise TypeError

        test_size = getattr(obj1[3], "test_size", None)
        if not test_size or not type(test_size) == float:
            logger.error('Corrupted dump. No feature parameters in the forth part.')
            raise TypeError

    return obj1


def restore_model(filepath):
    logger = logging.getLogger(__name__)

    with open(filepath, 'rb') as fileobj:
        obj1 = pickle.load(fileobj, encoding="latin1")

    obj1 = obj1[0]

    predict = getattr(obj1, "predict", None)
    if not predict or not callable(predict):
        logger.error('Corrupted dump. No model there.')
        raise TypeError

    return obj1