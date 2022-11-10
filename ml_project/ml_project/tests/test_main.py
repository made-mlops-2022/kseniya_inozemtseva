import os
import glob
import tempfile
import unittest
from unittest import mock
from io import StringIO

from ml_project.tests.synthetic_data import create_data_like
from ml_project.main import train, load_predict
from ml_project.params import tune_logging, get_project_root


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tune_logging(debug=False, stream=True)
        os.chdir(get_project_root().parent)

    def testTrain(self):

        for config_path in glob.glob('.\configs\*simple.json', recursive=False):

            with unittest.mock.patch('builtins.print') as m_output:
                train(config_path)
                last_msg = m_output.mock_calls[-1].args[0].split()

                self.assertEqual(['Saved', 'to'], last_msg[:-1])

                self.assertTrue(os.path.exists(last_msg[-1]))


class TestTrainPredict(unittest.TestCase):
    synt_dataset = None
    target_col = None
    dataset_path = None
    file = None
    test_len = 100

    @classmethod
    def setUpClass(cls):
        os.chdir(get_project_root().parent)
        synt_dataset = create_data_like(r".\data\raw\heart_cleveland_upload.csv", 'condition', cls.test_len)

        file, dataset_path = tempfile.mkstemp()
        synt_dataset.drop(columns=['condition']).to_csv(dataset_path, index=False)
        cls.file = file
        cls.dataset_path = dataset_path
        tune_logging(debug=False, stream=True)

    def testPredict(self):
        dataset_file = self.__class__.dataset_path
        model_file = glob.glob('.\models\*', recursive=False)[0]
        file, file_path = tempfile.mkstemp()
        try:
            load_predict([dataset_file,
                         model_file,
                          "--output_path", file_path,
                          "--stream"])
        except SystemExit as e:
            if not e.code:
                pass
            else:
                raise SystemExit(e)
        self.assertTrue(os.path.exists(file_path))
        os.close(file)
        with open(file_path, 'r') as file:
            res = list(map(lambda x: int(x.strip()), file.readlines()))
            self.assertLessEqual(sum(res), self.__class__.test_len / 1.5)
            self.assertLessEqual(self.__class__.test_len / 10, sum(res))

        os.remove(file_path)

    @classmethod
    def tearDown(cls) -> None:
        os.close(cls.file)
        os.remove(cls.dataset_path)
