# IMANAGE ENVIRONNEMENT
import os
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.optimizer_v2 import adam
from facematcher.siamese_neural_network.build.build_model import siamese_model
from facematcher.data_collection.preprocessing import preprocess


def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


# Import de the model
siamese_model = siamese_model()

# Setup Loss and Optimizer
binary_cross_loss = BinaryCrossentropy()
opt = adam.Adam(learning_rate=1e-4)

# Establish Checkpoints to periodically record
# and save weights and model parameters during training
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# The decorator is used to convert the function below
# into a TensorFlow computation graph. This optimizes function
# execution using TensorFlow's low-level features.
@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positives/negatives images
        X = batch[:2]
        # Get label
        y = batch[2]

        # Train the model (forward pass)
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


def train(data, epochs):

    print()
    print('The model is in training...')
    print()

    # Loop through epochs
    for epoch in range(1, epochs+1):
        print('\n Epoch {}/{}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # train step
            train_step(batch)
            progbar.update(idx+1)

        # Save checkpoints every 10 epochs
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
