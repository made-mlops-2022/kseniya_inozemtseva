import os
import unittest

from ml_project.params import *
from ml_project.features import *
from ml_project.features.features_utils import CustomTransformerCLevHeartDisease
from ml_project.tests.synthetic_data import create_data_like


class TestFeatures(unittest.TestCase):
    dataset = None
    feature_params = None

    @classmethod
    def setUpClass(cls):
        os.chdir(get_project_root().parent)
        cls.feature_params = FeatureParams()  # default values ok
        cls.dataset = create_data_like(r".\data\raw\heart_cleveland_upload.csv",
                                       cls.feature_params.target_col,
                                       100)

    def test_get_preprocessor(self):
        preprocessor = get_preprocessor(self.__class__.feature_params)

        self.assertTrue(hasattr(preprocessor, "transform"))

    def test_preprocess_scale(self):
        preprocessor = get_preprocessor(self.__class__.feature_params)
        dataset = preprocessor.fit_transform(self.__class__.dataset)

        self.assertEqual(self.__class__.dataset['oldpeak'].max() // 0.4, dataset['oldpeak'].max())
        self.assertEqual(self.__class__.dataset['trestbps'].max() // 10, dataset['trestbps'].max())

    def test_clev_heart_disease_transform1(self):
        transformer = CustomTransformerCLevHeartDisease(self.__class__.feature_params)
        df = self.__class__.dataset.copy()
        target = df[self.__class__.feature_params.target_col]
        df = df.drop(self.__class__.feature_params.target_col, axis=1)
        df_transformed = transformer.fit_transform(df, target)
        self.assertEqual(3 + len(df.columns), len(df_transformed.columns))

    def test_clev_heart_disease_transform2(self):
        transformer = CustomTransformerCLevHeartDisease(self.__class__.feature_params)
        df = self.__class__.dataset.copy()
        target = df[self.__class__.feature_params.target_col]
        df = df.drop(self.__class__.feature_params.target_col, axis=1)
        df_transformed = transformer.fit_transform(df, target)
        self.assertTrue('thal_oldpeak2' in df_transformed.columns)
        self.assertTrue('thal_trestbps2' in df_transformed.columns)
        self.assertTrue('sex_oldpeak2' in df_transformed.columns)

    def test_clev_heart_disease_transform3(self):
        transformer = CustomTransformerCLevHeartDisease(self.__class__.feature_params)
        df = self.__class__.dataset.copy()
        target = df[self.__class__.feature_params.target_col]
        df = df.drop(self.__class__.feature_params.target_col, axis=1)
        df_transformed = transformer.fit_transform(df, target)
        self.assertEqual(len(df_transformed), len(df))

    def test_clev_heart_disease_transform4(self):
        transformer = CustomTransformerCLevHeartDisease(self.__class__.feature_params)
        df = self.__class__.dataset.copy()
        target = df[self.__class__.feature_params.target_col]
        df = df.drop(self.__class__.feature_params.target_col, axis=1)
        df_transformed = transformer.fit_transform(df, target)
        col_li = [col for col in df.columns if col not in ('age', 'oldpeak', 'trestbps')]
        self.assertTrue(df[col_li].equals(df_transformed.loc[:, col_li]))

    def test_clev_heart_disease_transform5(self):
        transformer = CustomTransformerCLevHeartDisease(self.__class__.feature_params)
        df = self.__class__.dataset.copy()
        target = df[self.__class__.feature_params.target_col]
        df = df.drop(self.__class__.feature_params.target_col, axis=1)
        df_transformed = transformer.fit_transform(df, target)
        df1 = df_transformed['sex_oldpeak2']
        df2 = df_transformed['sex'].astype('str') + \
              "_" + \
              (df_transformed['oldpeak']).astype('int').astype('str')
        self.assertTrue(df1.equals(df2))

