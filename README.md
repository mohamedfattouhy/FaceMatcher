# FaceMatcher

Project description: This project is an enhanced version of the FaceRecognition repository by [nicknochnack](https://github.com/nicknochnack/FaceRecognition/tree/main). It aims to refactor and improve the code base by adding additional structure, comments and explanations about building machine learning models using TensorFlow and Keras. This project gave me the opportunity to learn how to create a complete machine learning model with Tensorflow and Keras.

## Installation

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/mohamedfattouhy/FaceMatcher.git`
2. Install the required dependencies: `pip install -r requirements.txt`


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

If the `main` file is executed, the following steps take place:

1. The `data` directory and the `anchor`, `positive` and `negative` sub-directories will be created.
2. You'll be asked to capture images of your face to be placed in directories `anchor` and `positive`.
3. Various face images will be downloaded from an image database and placed in the directory `negative`.
4. Then all these images will be concatenated and labeled (0: different, 1: same), and a train/test partition will be performed.
5. A `Siamese neural network` model will then be trained on the images using `TensorFlow` and `Keras` and saved.
6. The model is then used to perform real-time facial recognition with your webcam.
