# MANAGE ENVIRONNEMENT
import tensorflow as tf
from keras.models import Model
from keras.layers import (Layer,
                          Conv2D,
                          Dense,
                          MaxPooling2D,
                          Input,
                          Flatten)


# L1 Distance class
class L1Distance(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()  # executes the initialization method of the parent Layer class

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        # Compute L1 distance
        return tf.math.abs(input_embedding - validation_embedding)


def make_embedding() -> Model:

    # Input (image of size 100x100px)
    input = Input(shape=(100, 100, 3), name='input_image')

    # First block
    conv_1 = Conv2D(64, (10, 10), activation='relu')(input)
    max_poo1ing_1 = MaxPooling2D(64, (2, 2), padding='same')(conv_1)

    # Second block
    conv_2 = Conv2D(128, (7, 7), activation='relu')(max_poo1ing_1)
    max_poo1ing_2 = MaxPooling2D(64, (2, 2), padding='same')(conv_2)

    # Third block
    conv_3 = Conv2D(128, (4, 4), activation='relu')(max_poo1ing_2)
    max_pooling_3 = MaxPooling2D(64, (2, 2), padding='same')(conv_3)

    # Final embedding block
    conv_4 = Conv2D(256, (4, 4), activation='relu')(max_pooling_3)
    flatten = Flatten()(conv_4)
    dense = Dense(4096, activation='sigmoid')(flatten)

    return Model(inputs=[input], outputs=[dense], name='embedding')


def siamese_model() -> Model:

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image input in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    embedding = make_embedding()

    # Combine siamese distance components
    siamese_layer = L1Distance()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image),
                              embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image],
                 outputs=classifier, name='SiameseNetwork')


# embedding_model = make_siamese_model()
# print(embedding_model.summary())
