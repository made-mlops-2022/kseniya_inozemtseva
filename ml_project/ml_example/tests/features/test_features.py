from ml_example.data import *
from ml_example.params import *

import unittest


class TestFeatures(unittest.TestCase):
    def test_get_processor(self):
        filelike_obj = StringIO()
        data = [['Spam'] * 5 + ['Baked Beans'],
                ['Spam', 'Lovely Spam', 'Wonderful Spam']]
        dump_data(data, filelike_obj, writer=CsvWriter())

        filelike_obj.seek(0)
        self.assertEqual(filelike_obj.readline(), ','.join(data[0]) + '\r\n')

        dataset_path =
        # target_col: str
        data = read_dataset(dataset_path)
        self.assertGreater(len(data), 10)
        #assert target_col in data.columns

    def test_get_processor(self):
        filelike_obj = StringIO()
        data = [['Spam'] * 5 + ['Baked Beans'],
                ['Spam', 'Lovely Spam', 'Wonderful Spam']]
        dump_data(data, filelike_obj, writer=CsvWriter())

        filelike_obj.seek(0)
        self.assertEqual(filelike_obj.readline(), ','.join(data[0]) + '\r\n')

        dataset_path =
        # target_col: str
        data = read_dataset(dataset_path)
        self.assertGreater(len(data), 10)
        #assert target_col in data.columns

