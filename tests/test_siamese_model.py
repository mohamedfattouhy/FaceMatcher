#  MANAGEMENT ENVIRONMENT --------------------------------
import os
import unittest
import tensorflow as tf
from keras.models import Model
from facematcher.data_collection.preprocessing import create_dataset, preprocess_twin
from facematcher.siamese_neural_network.build.build_model import make_embedding, siamese_model
from facematcher.siamese_neural_network.train.train_model import train


# TESTS  -------------------------------
class TestSiamseseModel(unittest.TestCase):

    def setUp(self):

        self.ANC_PATH = os.path.join('tests', 'data_test', 'anchor')
        self.POS_PATH = os.path.join('tests', 'data_test', 'positive')
        self.NEG_PATH = os.path.join('tests', 'data_test', 'negative')

        self.anchor_dataset = tf.data.Dataset.list_files(os.path.join(self.ANC_PATH, r'*.jpg'))
        self.positive_dataset = tf.data.Dataset.list_files(os.path.join(self.POS_PATH, r'*.jpg'))
        self.negative_dataset = tf.data.Dataset.list_files(os.path.join(self.NEG_PATH, r'*.jpg'))
    
        self.data = create_dataset(self.anchor_dataset,
                                   self.positive_dataset,
                                   self.negative_dataset)

        self.data = self.data.map(preprocess_twin)
        self.data = self.data.cache()
        self.data = self.data.shuffle(buffer_size=10)
        self.data = self.data.batch(1)
        self.data = self.data.prefetch(1)

    def test_model_type(self):

        embedding = make_embedding()
        model = siamese_model()

        self.assertIsInstance(embedding, Model)
        self.assertIsInstance(model, Model)

    def test_train_model(self):
        try:
            train(data=self.data, epochs=1)
        except Exception as e:
            self.fail(f"Error: {e}")


if __name__ == "__main__":
    unittest.main()
