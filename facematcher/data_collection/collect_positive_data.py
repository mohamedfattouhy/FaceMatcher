# IMANAGE ENVIRONNEMENT
import os
import uuid
import cv2


def capture_positive_images(anchor_path, positvie_path):

    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)  # Don't hesitate to try several values (1,2,...) if your webcam don't turn on

    print()
    print("Press \'a\' to capture anchor image \n"
          "Press \'p\' to capture positive image \n"
          "Press \'q\' to quit")
    print()

    while cap.isOpened():
        _, frame = cap.read()

        # Cut down frame to 250x250px
        frame = frame[120:120+250, 200:200+250, :]

        # Collect anchors images by pressing 'a'
        if cv2.waitKey(10) & 0XFF == ord('a'):

            # Create unique file path (uuid is used to generate unique image name)
            imgname = os.path.join(anchor_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives by pressing 'p'
        if cv2.waitKey(10) & 0XFF == ord('p'):

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
