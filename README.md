# FaceMatcher

Project description: This project is an enhanced version of the FaceRecognition repository by [nicknochnack](https://github.com/nicknochnack/FaceRecognition/tree/main). It aims to refactor and improve the code base by adding additional structure, comments and explanations about building machine learning models using TensorFlow and Keras. This project gave me the opportunity to learn how to create a complete machine learning model with Tensorflow and Keras.

## Siamese Neural Networks for Face Recognition

This project leverages Siamese neural networks to perform face recognition. The network architecture is designed to compare and identify similarities between pairs of images. Specifically, it focuses on three types of images: anchors, positives, and negatives.

- **Anchors**: Anchors represent the reference images or known identities. These are the images against which other images will be compared to determine if they belong to the same person.

- **Positives**: Positive images are variations of the anchor images and should be recognized as the same person by the network. These images may include different poses, lighting conditions, or facial expressions.

- **Negatives**: Negative images, on the other hand, depict different individuals faces. The network is trained to understand that these images do not match the anchor images and should be distinguished as different individuals.

Once the model is trained, we use a **verification** directory containing images from the **positives** directory. Each time an image is given as input to the model (using the webcam, for example), this image is compared with all the images in the **verification** directory, and a verification threshold is set above which we declare whether the input image is indeed the person in question.

## Installation

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/mohamedfattouhy/FaceMatcher.git`
2. Install the required dependencies: `pip install -r requirements.txt`

If you prefer, you can also install the `facematcher` package like this: `pip install git+https://github.com/mohamedfattouhy/FaceMatcher`

## Project Structure

The project is structured as follows:  

- `data/`:  Automatically created directory that will contains the dataset used for training.
- `facematcher/data_cllection/`: Contains scripts for data collection and pre-processing.
- `facematcher/siamese_neural_network`: Includes the machine learning models for face recognition.
- `facematcher/facial_recognition`: Contains a script to enable the model to be used in real time for facial recognition
- `main.py`: This file is used to carry out all the steps required to test the facial recognition model in real-time.

## Getting Started

1. Navigate to the project directory: `cd path/to/FaceMatcher`
2. Run the following command line to be able to use a real-time facial recognition test: `python main.py` 


## Usage

If the **main** file is executed, the following steps take place:

1. The **data** directory and the **anchor**, **positive** and **negative** sub-directories will be created.
2. You'll be asked to capture images of your face to be placed in directories **anchor** and **positive**.
3. Various face images will be downloaded from an image database and placed in the directory **negative**.
4. Then all these images will be concatenated and labeled (0: different, 1: same), and a train/test partition will be performed.
5. A **Siamese neural network** model will then be trained on the images using `TensorFlow` and `Keras` and saved.
6. The model is then used to perform real-time facial recognition with your webcam.
