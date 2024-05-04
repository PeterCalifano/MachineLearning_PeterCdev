# Script implementing tutorials in TensorFlow documentation to show the basics.
# Created by PeterC - 29-04-2024
# 1) https://www.tensorflow.org/tutorials/quickstart/beginner --> MNIST classified

# Import modules
import tensorflow as tf
from tensorflow import keras
import numpy as np
print("TensorFlow version:", tf.__version__)

# %% Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


print('Input X train shape:', x_train.shape) 
# Note the shape required by Tensorflow: [Nsamples, sampleDim1, sampleDim2] --> matches input layer shape
# Flatten layer automatically flatten the input data into a 1D vector for convenience
print('Labels Y train shape:', y_train.shape)

# Normalize input features data to [0,1] (type is uint8)
x_train, x_test = x_train / 255.0, x_test / 255.0

# %% Define model architecture
sizeH1 = 128 # Number of neurons in 1st hidden layer

MNIST_classifier = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Input layer taking the 28x28 pixels MNIST image
    keras.layers.Dense(sizeH1, activation='relu'), # Define 1st hidden layer
    keras.layers.Dropout(0.2), # Dropout layer
    keras.layers.Dense(10) # Output layer --> 10 classes
    ])

# Test forward prediction by evaluating model with first input
prediction = MNIST_classifier(x_train[:1]).numpy() # Converts from TF tensor to Numpy array
print('Prediction example (logits):',prediction)

# Convert output to probabilities using Softmax
print('Prediction example (softmax):',tf.nn.softmax(prediction).numpy())

# Define loss function for training (Sparse categorial Cross Entropy is good for classification)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# %% Compile model setting optimizer, loss function and metric
MNIST_classifier.compile(optimizer='adam',
                        loss=loss_fn,
                        metrics=['accuracy'])

# Execute training using model.fit function
MNIST_classifier.fit(x_train, y_train, epochs=5)

# Evaluate the model using evaluate method on validation dataset
MNIST_classifier.evaluate(x_test,  y_test, verbose=2)

# Optionally define a wrapper for the model including the softmax function (not trained)
MNIST_ProbabilityClassifier = tf.keras.Sequential([
  MNIST_classifier,
  tf.keras.layers.Softmax()
])

# Evaluate the probabilistic classifier with random inputs(not with evaluate)
MNIST_ProbabilityClassifier(x_test[:5])
