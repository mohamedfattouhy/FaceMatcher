# MANAGE ENVIRONNEMENT
import os
import tensorflow as tf
from keras.metrics import Precision, Recall
from facematcher.data_collection.create_folder import create_folder
from facematcher.facial_recognition.webcam_real_time_test import real_time_facial_recognition
from facematcher.data_collection.collect_positive_data import capture_positive_images
from facematcher.data_collection.load_negative_data import uncompress_and_move_lfw_dataset
from facematcher.data_collection.preprocessing import create_dataset, generate_new_facial_images
from facematcher.data_collection.preprocessing import preprocess_twin
from facematcher.siamese_neural_network.build.build_model import L1Distance
from facematcher.siamese_neural_network.train.train_model import train
from facematcher.siamese_neural_network.build.build_model import siamese_model


# create 'data' folder if it does not already exist
# and "anchor", "positive", "negative" sub-folders
create_folder(dirpath_name='data',
              subdir_names=["anchor", "positive", "negative"])

# # Setup paths to data
ANC_PATH = os.path.join('data', 'anchor')
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')

# Load positives images
capture_positive_images(anchor_path=ANC_PATH, positvie_path=POS_PATH)

# Load negatives images from http://vis-www.cs.umass.edu/lfw/#download
uncompress_and_move_lfw_dataset()

anchor_dataset = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg')
positive_dataset = tf.data.Dataset.list_files(POS_PATH+'\*.jpg')
negative_dataset = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg')

data = create_dataset(anchor_dataset, positive_dataset, negative_dataset)

samples = data.as_numpy_iterator()
examples = samples.next()
print()
# print("Sample from data: ", examples)  # uncomment to see the output

# Avoid out of memory errors by setting GPU Memory Consumption Growth
# If your machine has no GPU, this code will have no effect
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

print()
print(f'Number of gpu found on your machine: {len(gpus)}')
print()

# # Uncomment the following lines if you want to
# # generate new images from the originals
# # to increase the size of the image base

# data_dir_aug = os.path.join('data', 'anchor')
# output_dir_aug = os.path.join('data', 'anchor')
# generate_new_facial_images(data_dir=data_dir_aug, output_dir=output_dir_aug)


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10_000)

# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# Train the model
EPOCHS = 50
train(train_data, epochs=EPOCHS)

# Import de the model
siamese_model = siamese_model()

# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
# Make predictions
y_hat = siamese_model.predict([test_input, test_val])
y_hat

# Creating a metric object
recall = Recall()
precision = Precision()

# Calculating the recall value
recall.update_state(y_true, y_hat)
precision.update_state(y_true, y_hat)

# Return Recall Result
print()
print("Recall: ", recall.result().numpy())
print("Precision: ", precision.result().numpy())
print()

# Save weights
path_save_model = os.path.join('facematcher', 'siamese_neural_network',
                               'model', 'siamesemodel.h5')
siamese_model.save(path_save_model)


# Load the trained model
model_path = os.path.join('facematcher', 'siamese_neural_network',
                          'model', 'siamesemodel.h5')
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'L1Dist': L1Distance}
                                   )

# Run the real-time facial recognition
if __name__ == '__main__':
    real_time_facial_recognition(model=model)
