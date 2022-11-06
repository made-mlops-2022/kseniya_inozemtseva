import os
import glob
import tempfile
import unittest
from unittest import mock
from io import StringIO

from ml_example.tests.synthetic_data import create_data_like
from ml_example.main import train, load_predict
from ml_example.params import tune_logging


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tune_logging(debug=False, stream=True)
        print(__file__)
        # TODO: dont do this
        os.chdir("..\\")

    def testTrain(self):

        for config_path in glob.glob('..\configs\*simple.json', recursive=False):
            #config_path = r"configs\cle_hea_diss_simple.json"

            with unittest.mock.patch('builtins.print') as m_output:
                train(config_path)
                last_msg = m_output.mock_calls[-1].args[0].split()

                self.assertEqual(['Saved', 'to'], last_msg[:-1])

                self.assertTrue(os.path.exists(last_msg[-1]))


class TestTrainPredict(unittest.TestCase):
    synt_dataset = None
    target_col = None
    dataset_path = None

    @classmethod
    def setUpClass(cls):
        os.chdir("..\\")
        synt_dataset = create_data_like(r"..\data\raw\heart_cleveland_upload.csv", 'condition', 100)
        #filelike = StringIO()
        #synt_dataset.to_csv(filelike, line_terminator=r'\r\n')
        #filelike.seek(0)
        #cls.dataset_file = filelike
        file, dataset_path = tempfile.mkstemp()
        synt_dataset.to_csv(dataset_path)
        cls.dataset_path = dataset_path
        tune_logging(debug=False, stream=True)

    def testPredict(self):
        dataset_file = self.__class__.dataset_path
        model_file = glob.glob('..\models\*', recursive=False)[0]
        load_predict(["data_path" ,dataset_file,
                     "model_path", model_file,
                      "--output_path", "output.txt",
                      "--stream"])
        self.assertTrue(True)

    @classmethod
    def tearDown(cls) -> None:
        os.remove(cls.dataset_path)
