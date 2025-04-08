# Handwritten Digit Classifier

This project implements a neural network to classify handwritten digits using the MNIST dataset with TensorFlow and Keras.

## Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), split into 60,000 training images and 10,000 testing images. Each image is 28x28 pixels. This project builds a simple neural network to classify these digits with high accuracy.

## Requirements

- TensorFlow
- Keras
- Matplotlib
- scikit-learn

## Dataset

The MNIST dataset is loaded directly from Keras using the built-in dataset loading functionality. The dataset provides:

- Training set: 60,000 images (28x28 pixels)
- Testing set: 10,000 images
- Labels: Digits from 0 to 9

## Data Preprocessing

1. Normalization: Pixel values are scaled from [0, 255] to [0, 1]
2. Flattening: Images are flattened from 28x28 matrices to 784-element vectors during model creation

## Model Architecture

A sequential model with the following layers:
- Input: Flattened 28x28 = 784 neurons
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 10 neurons (one for each digit) with softmax activation

Total parameters: 104,938

## Training

The model is trained with the following configuration:
- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Epochs: 15
- Validation Split: 20% of training data

## Performance

The model achieves:
- Training accuracy: ~99.7%
- Test accuracy: 97.72%

## How to Run the Code

1. Open the notebook in Google Colab or your local Jupyter environment
2. Run the import statements to ensure all required libraries are available
3. Load the MNIST dataset using the keras.datasets module
4. Run the preprocessing steps to normalize the pixel values
5. Create the model architecture with the sequential API and add the layers
6. Compile the model with the appropriate loss function, optimizer and metrics
7. Train the model by fitting it to the training data
8. Evaluate the model on the test data to check accuracy
9. Use the trained model to make predictions on new images

## Visualization

The notebook includes steps to visualize:
- Sample digits from the dataset
- Training and validation loss curves
- Training and validation accuracy curves

## Making Predictions

To predict a digit from a new image, reshape the image to match the input shape expected by the model and use the prediction method. The output with the highest probability indicates the predicted digit.

## Further Improvements

Potential enhancements could include:
- Adding convolutional layers to improve accuracy
- Implementing data augmentation
- Using dropout for regularization
- Experimenting with different architectures and hyperparameters
