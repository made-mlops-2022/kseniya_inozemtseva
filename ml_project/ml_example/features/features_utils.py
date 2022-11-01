from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin


class CustomTransformerCLevHeartDisease(TransformerMixin):
    def transform(self, X, y):
        X_ = X.copy()

        X_['thal_oldpeak2'] = X_['thal'].astype('str') + "_" + (X_['oldpeak'] // 0.4).astype('int').astype('str')
        X_['thal_trestbps2'] = X_['thal'].astype('str') + "_" + (X_['trestbps'] // 10).astype('str')
        X_['sex_oldpeak2'] = X_['sex'].astype('str') + "_" + (X_['oldpeak'] // 0.4).astype('int').astype('str')
        return X_


def get_processor(
        feature_params
):
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    simple_transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, feature_params.numerical_cols),
            ('cat', categorical_transformer, feature_params.categorical_cols)
        ], remainder='passthrough')

    custom_transformer = ColumnTransformer(
        transformers=[
            ('cust1', CustomTransformerCLevHeartDisease, []),
        ], remainder='passthrough')

    preproc = FeatureUnion([
        ('common_ops', simple_transformer),
        ('custom_ops', custom_transformer)
    ])

    processor = Pipeline(steps=[
        ('preproc', preproc)
    ])

    return processor

