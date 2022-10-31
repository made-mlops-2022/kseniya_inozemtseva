from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import get_scorer
from dataclasses import asdict


def get_model(
        pipe, model_params
):
    if model_params.type == 'RandomForest':
        if model_params.params_tunning:
            grid = GridSearchCV(make_pipeline(pipe, RandomForestClassifier(random_state=model_params.random_state)),
                                param_grid=asdict(model_params.params_grid),
                                cv=2,
                                scoring=model_params.metrics[0])

            return grid
        else:
            return RandomForestClassifier(random_state=model_params.random_state,
                                          **asdict(model_params.params))
    else:
        raise NotImplementedError


def score(predictions, true_values, score_func):
    return get_scorer(score_func)._score_func(
        true_values,
        predictions
    )
