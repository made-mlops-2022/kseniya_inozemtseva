from ml_example.data import *
from ml_example.params import *

import unittest


class TestData(unittest.TestCase):
    def test_load_dataset(self):
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

    def test_split_dataset(self):
        # tmpdir, dataset_path: str
        val_size = 0.2
        splitting_params = SplittingParams(random_state=239, val_size=val_size,)
        data = read_dataset(dataset_path)
        train, val = split_train_val_data(data, splitting_params)
        self.assertGreater(train.shape[0], 10)
        #assert val.shape[0] > 10


if __name__ == '__main__':
    test = TestData()