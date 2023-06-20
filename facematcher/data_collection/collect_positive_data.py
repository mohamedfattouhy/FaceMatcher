# IMANAGE ENVIRONNEMENT
import os
import uuid
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image_dataset import image_dataset_from_directory
from keras.preprocessing.image import save_img


def CapturePositiveImages(anchor_path, positvie_path):

    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)  # Don't hesitate to try several values (1,2,...) if your webcam don't turn on

    print()
    print("""Press \'a\' to capture anchor image \n
          Press \'p\' to capture positive image \n
          Press \'q\' to quit""")
    print()

    while cap.isOpened():
        _, frame = cap.read()

        # Cut down frame to 250x250px
        frame = frame[120:120+250, 200:200+250, :]

        # Collect anchors images by pressing 'a'
        if cv2.waitKey(1) & 0XFF == ord('a'):

            # Create unique file path (uuid is used to generate unique image name)
            imgname = os.path.join(anchor_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives by pressing 'p'
        if cv2.waitKey(1) & 0XFF == ord('p'):

            # Create the unique file path
            imgname = os.path.join(positvie_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)

        # Show image back to screen
        cv2.imshow('Image Collection', frame)

        # Stop session by pressing 'q'
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


def generate_new_facial_images(data_dir, output_dir, n_new_images=10):

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
