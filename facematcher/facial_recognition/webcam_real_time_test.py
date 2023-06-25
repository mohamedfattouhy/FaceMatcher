# MANAGE ENVIRONNEMENT
import os
import shutil
import cv2
import numpy as np
from facematcher.data_collection.preprocessing import preprocess
from facematcher.data_collection.create_folder import create_folder


# Verification function
def verify(model, detection_threshold: float,
           verification_threshold: float) -> tuple:
    """checks whether an input image matches the person for whom the model has been trained to recognize

    Args:
        model: trained machine learning model

        detection_threshold (float): detection threshold (between 0 and 1), where detection is a
        number of positive predictions (similarity > detection_threshold)

        verification_threshold (float): verification threshold (between 0 and 1), where verification is
        the proportion of positives predictions

    Returns:
        tuple (length 2): list of similarities and a boolean resulting from the comparison
        between the proportion of positives predictions and verification_threshold

    >>> model_path = os.path.join('facematcher', 'siamese_neural_network',
                          'model', 'siamesemodel.h5')
        load_model = tf.keras.models.load_model(model_path)
        results, verified = verify(load_model, 0.9, 0.7)
    """

    if not (isinstance(detection_threshold, float) and
            isinstance(verification_threshold, float)):
        raise TypeError("The threshold parameter must be of type float.")

    if (detection_threshold < 0 or detection_threshold > 1) or\
       (verification_threshold < 0 or verification_threshold > 1):
        raise ValueError("The threshold parameter must be between 0 and 1.")

    # Build results list
    results = []

    # set path to the verification dir
    validation_dir_path = os.listdir(os.path.join('application_data',
                                                  'verification_images'))

    input_img_path = os.path.join('application_data',
                                  'input_image', 'input_image.jpg')
    input_img = preprocess(input_img_path)

    for image in validation_dir_path:

        verification_img_path = os.path.join('application_data',
                                             'verification_images', image)
        validation_img = preprocess(verification_img_path)

        # Make prediction (0: diffrent, 1: same)
        result = model.predict(list(np.expand_dims([input_img,
                                                    validation_img], axis=1))
                               )
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of true positive predictions
    verification = detection / len(os.listdir(os.path.join('application_data',
                                                           'verification_images')))

    # verfied is True if verification is greater than verification_threshold
    verified = verification > verification_threshold

    return results, verified


def real_time_facial_recognition(model) -> None:
    """tests a trained model using real-time input images from the webcam

    Args:
        model: trained machine learning model

    Returns:
        None

    >>> model_path = os.path.join('facematcher', 'siamese_neural_network',
                          'model', 'siamesemodel.h5')
        load_model = tf.keras.models.load_model(model_path)
        real_time_facial_recognition(model=load_model)
    """

    # create the directories required to store images for real-time testing
    create_folder('application_data', ['input_image', 'verification_images'])

    # we copy all positves images to application_data/verification_images
    EX_PATH = os.path.join('data', 'positive')
    NEW_PATH = os.path.join('application_data', 'verification_images')
    shutil.copytree(EX_PATH, NEW_PATH, dirs_exist_ok=True)

    # Webcam connection
    cap = cv2.VideoCapture(0) # Don't hesitate to try several values (1,2,...) if your webcam don't turn on
    print()
    print('Press \'i\' to capture an input image and press \'q\' to quit')
    print()

    while cap.isOpened():

        _, frame = cap.read()
        frame = frame[120:120+250, 200:200+250, :]

        cv2.imshow('Capture', frame)

        # Input image trigger (by pressing 'i')
        if cv2.waitKey(10) & 0xFF == ord('i'):

            # Save input image to application_data/input_image/ folder
            cv2.imwrite(os.path.join('application_data', 'input_image',
                                     'input_image.jpg'), frame)

            # Run verification
            _, verified = verify(model, 0.9, 0.7)
            print('Verified: ', verified)

        # Stop session by pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
