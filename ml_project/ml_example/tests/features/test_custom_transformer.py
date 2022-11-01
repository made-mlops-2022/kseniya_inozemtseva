import unittest
import pandas as pd

from ml_example.data import *
from ml_example.params import *
from ml_example.features.features_utils import CustomTransformerCLevHeartDisease

from ml_example.tests.synthetic_data import create_data_like


class TestCustomTransformer(unittest.TestCase):
    synt_dataset = None
    target_col = None

    @classmethod
    def setUpClass(cls):
        cls.target_col = 'condition'
        cls.synt_dataset = create_data_like(r"..\..\..\data\raw\heart_cleveland_upload.csv", cls.target_col, 100)

    def test_clev_heart_disease_transform1(self):
        transformer = CustomTransformerCLevHeartDisease()
        df = self.__class__.synt_dataset.copy()
        target = df[self.__class__.target_col]
        df = df.drop(self.__class__.target_col, axis=1)
        df_transformed = transformer.transform(df, target)
        self.assertEqual(3 + len(df.columns), len(df_transformed.columns))

    def test_clev_heart_disease_transform2(self):
        transformer = CustomTransformerCLevHeartDisease()
        df = self.__class__.synt_dataset.copy()
        target = df[self.__class__.target_col]
        df = df.drop(self.__class__.target_col, axis=1)
        df_transformed = transformer.transform(df, target)
        self.assertTrue('thal_oldpeak2' in df_transformed.columns)
        self.assertTrue('thal_trestbps2' in df_transformed.columns)
        self.assertTrue('sex_oldpeak2' in df_transformed.columns)

    def test_clev_heart_disease_transform3(self):
        transformer = CustomTransformerCLevHeartDisease()
        df = self.__class__.synt_dataset.copy()
        target = df[self.__class__.target_col]
        df = df.drop(self.__class__.target_col, axis=1)
        df_transformed = transformer.transform(df, target)
        self.assertEqual(len(df_transformed), len(df))

    def test_clev_heart_disease_transform4(self):
        transformer = CustomTransformerCLevHeartDisease()
        df = self.__class__.synt_dataset.copy()
        target = df[self.__class__.target_col]
        df = df.drop(self.__class__.target_col, axis=1)
        df_transformed = transformer.transform(df, target)

        self.assertTrue(df.equals(df_transformed.loc[:, list(df.columns)]))

    def test_clev_heart_disease_transform5(self):
        transformer = CustomTransformerCLevHeartDisease()
        df = self.__class__.synt_dataset.copy()
        target = df[self.__class__.target_col]
        df = df.drop(self.__class__.target_col, axis=1)
        df_transformed = transformer.transform(df, target)
        df1 = df_transformed['sex_oldpeak2']
        df2 = df_transformed['sex'].astype('str') + \
              "_" + \
              (df_transformed['oldpeak'] // 0.4).astype('int').astype('str')
        self.assertTrue(df1.equals(df2))
