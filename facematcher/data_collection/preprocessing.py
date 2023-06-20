# IMANAGE ENVIRONNEMENT
import os
import matplotlib.pyplot as plt
import tensorflow as tf


def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)

    # Decode a JPEG image to a uint8 tensor
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))

    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return pre-processed image
    return img


# set to True to see an example of before/after pre-processed images
display_images = False

if display_images:

    # path to positive images
    POS_PATH = os.path.join('..', '..', 'data', 'positive')

    # Retrieve the full path of the first positive image
    files = os.listdir(POS_PATH)
    path_to_image = os.path.join(POS_PATH, files[0])

    byte_img = tf.io.read_file(path_to_image)
    img_before_preprocess = tf.io.decode_jpeg(byte_img)

    img_after_preprocess = preprocess(path_to_image)

    # Create a figure with two subgraphs
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(img_before_preprocess)
    axs[1].imshow(img_after_preprocess)

    # Adjust spacing between subgraphs
    plt.subplots_adjust(wspace=0.2)

    # Display figure
    plt.show()


# the zip method will allow us to associate references
# images (anchors) and positives or negatives images with
# the corresponding label (0: same, 1: different)
def create_dataset(anchor_dataset, positive_dataset, negative_dataset):

    positives = tf.data.Dataset.zip((anchor_dataset, positive_dataset,
                                    tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor_dataset)))
                                    ))
    negatives = tf.data.Dataset.zip((anchor_dataset, negative_dataset,
                                    tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor_dataset)))
                                    ))

    # concatenate positves and negatives images in one dataset
    dataset = positives.concatenate(negatives)

    return dataset
