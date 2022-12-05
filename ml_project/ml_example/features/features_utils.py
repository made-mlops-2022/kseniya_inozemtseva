from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


class CustomTransformerCLevHeartDisease(BaseEstimator, TransformerMixin):
    def __init__(self, feature_params):
        super().__init__()
        self.params = feature_params
        self.scaler = StandardScaler()

    def transform(self, X_, y=None):
        X = X_.copy()
        X[self.params.cols_to_std_scale] = self.scaler.transform(X[self.params.cols_to_std_scale])
        for col_name, factor in zip(*self.params.scale_params):
            X.loc[:, [col_name]] = X.loc[:, [col_name]] // factor
        X['thal_oldpeak2'] = X['thal'].astype('str') + "_" + (X['oldpeak']).astype('int').astype('str')
        X['thal_trestbps2'] = X['thal'].astype('str') + "_" + (X['trestbps']).astype('str')
        X['sex_oldpeak2'] = X['sex'].astype('str') + "_" + (X['oldpeak']).astype('int').astype('str')
        return X

    def fit(self, X, y=None):
        self.scaler.fit(X[self.params.cols_to_std_scale])
        return self


def get_preprocessor(
        feature_params
):
    preprocessor = Pipeline(steps=[
        ('custom_ops', CustomTransformerCLevHeartDisease(feature_params))
    ])

    return preprocessor
