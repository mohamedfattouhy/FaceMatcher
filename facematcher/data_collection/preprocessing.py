# MANAGE ENVIRONNEMENT
import os
import uuid
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image_dataset import image_dataset_from_directory
from keras.preprocessing.image import save_img


def preprocess(file_path: str) -> tf.image:
    """pre-process images by resizing them to 100x100px and normalizing them

    Args:
        file_path (str): path to the image

    Returns:
        tf.image: the preprocessed image
    """

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


def preprocess_twin(input_img: str, validation_img, label: int) -> tuple:
    """pre-process a pair of images.

    Args:
        input_img (str): path to the input image (anchor)
        validation_img (str): path to the validation image (positive or negative)
        label (int): 0 (different) or 1 (same)

    Returns:
        tuple (length 3): the pre-processed images and associated label
    """
    return (preprocess(input_img), preprocess(validation_img), label)


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
    """create a concatenated dataset from several datasets

    Args:
        anchor_dataset (dataset): dataset containing anchors images
        positive_dataset (dataset): dataset containing positives images
        negative_dataset (dataset): dataset containing negatives images

    Returns:
        dataset: the concatenated dataset
    """

    positives = tf.data.Dataset.zip((anchor_dataset, positive_dataset,
                                    tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor_dataset)))
                                    ))
    negatives = tf.data.Dataset.zip((anchor_dataset, negative_dataset,
                                    tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor_dataset)))
                                    ))

    # concatenate positves and negatives images in one dataset
    dataset = positives.concatenate(negatives)

    print()
    print("The dataset has been created")
    print()

    return dataset


def generate_new_facial_images(data_dir: str, output_dir: str,
                               n_new_images: int = 10):
    """create new images from the original ones

    Args:
        data_dir (str): path to load original images from
        output_dir (str): path to save the new images
        n_new_images (int): number of images to create, default is 10

    Returns:
        None

    >>> data_dir_path = os.path.join('data', 'anchor')
        output_dir_path = os.path.join('data', 'anchor')
        generate_new_facial_images(data_dir=data_dir_path, output_dir=output_dir_path)
    """

    # generate new images from the originals
    datagen = ImageDataGenerator(
        # Random image rotation in the range -20 to 20 degrees
        rotation_range=20,
        # Random horizontal image shift within the range -0.1 to 0.1 of total width
        width_shift_range=0.1,
        # Random vertical shift of the image in the range -0.1 to 0.1 of the total height
        height_shift_range=0.1,
        # Random horizontal flipping of the image
        horizontal_flip=True
    )

    # Load existing images
    image_data = image_dataset_from_directory(
        data_dir,
        labels=None,
        # Resize images to 100x100px
        image_size=(100, 100),
        # Use a batch size of 1 to generate one image at a time
        batch_size=1,
        shuffle=True
    )

    for images in image_data:
        for new_image in range(1, n_new_images):

            # Apply data augmentation transformations
            new_image_aug = datagen.flow(images, batch_size=1)
            new_image_reshape = new_image_aug[0].reshape(100, 100, 3)

            filename = f'augmented_image_{uuid.uuid1()}.jpg'
            # Save the augmented images
            save_img(output_dir + '/' + filename, new_image_reshape)

    print()
    print("New images have been created")
    print()
