#  MANAGEMENT ENVIRONMENT --------------------------------
import os
import unittest
import tensorflow as tf
from facematcher.data_collection.preprocessing import preprocess, preprocess_twin
from facematcher.data_collection.preprocessing import create_dataset


# TESTS  -------------------------------
class PreprocessImages(unittest.TestCase):

    def setUp(self):
        self.img_path = os.path.join('tests', 'data_test', 'negative',
                                     'Aaron_Peirsol_0001.jpg')

        self.ANC_PATH = os.path.join('tests', 'data_test', 'anchor')
        self.POS_PATH = os.path.join('tests', 'data_test', 'positive')
        self.NEG_PATH = os.path.join('tests', 'data_test', 'negative')

        self.anchor_dataset = tf.data.Dataset.list_files(os.path.join(self.ANC_PATH, r'*.jpg'))
        self.positive_dataset = tf.data.Dataset.list_files(os.path.join(self.POS_PATH, r'*.jpg'))
        self.negative_dataset = tf.data.Dataset.list_files(os.path.join(self.NEG_PATH, r'*.jpg'))

        self.data = create_dataset(self.anchor_dataset,
                                   self.positive_dataset,
                                   self.negative_dataset)

    def test_preprocess(self):

        img = preprocess(file_path=self.img_path)
        self.assertEqual(img.shape, (100, 100, 3))
        self.assertIsInstance(img, tf.Tensor)
        self.assertTrue((img.numpy().max() >= 0) and (img.numpy().max() <= 1))

        sample_1 = self.data.as_numpy_iterator().next()
        self.assertIsInstance(sample_1, tuple)
        self.assertEqual(len(sample_1), 3)

        self.data_preprocessed_twin = self.data.map(preprocess_twin)
        samples_2 = self.data_preprocessed_twin.as_numpy_iterator().next()
        self.assertIsInstance(samples_2, tuple)
        self.assertEqual(len(samples_2), 3)


# if __name__ == "__main__":
#     unittest.main()
