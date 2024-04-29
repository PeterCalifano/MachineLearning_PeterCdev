# Script implementing tutorials in TensorFlow documentation to show the basics.
# Created by PeterC - 29-04-2024
# 1) https://www.tensorflow.org/tutorials/quickstart/advanced

# Import modules
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import numpy as np
print("TensorFlow version:", tf.__version__)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add new dimension to the problem --> perform batch training
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# Create a dataset of mini-batches and performing shuffling
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
# Shuffle test data as well
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

print('Shape of train_ds:', train_ds)
print('Shape of test_ds:', test_ds)

print('Input X train shape:', x_train.shape) 
# Note the shape required by Tensorflow: [Nsamples, sampleDim1, sampleDim2] --> matches input layer shape
# Flatten layer automatically flatten the input data into a 1D vector for convenience
print('Labels Y train shape:', y_train.shape)

# Normalize input features data to [0,1] (type is uint8)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define custom architecture with standard Tensorflow layers
class MyModel(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

# Define loss function object (pre-defined in TF)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define Adam optimizer object with default options
optimizer = tf.keras.optimizers.Adam()

# Define metrics instances for training and validation
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Manually specify code to train the model by using GradientTape module
@tf.function # Defines a TensorFlow function --> single training step
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    # FORWARD PASS --> evaluate loss function of tf the model
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

    # BACKWARD PASS --> compute gradients of the model (trainable only) parameters 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Apply gradients to compute updated values of the parameters

    # Evaluate training metrics 
    train_loss(loss)
    train_accuracy(labels, predictions)

# Specify code for validation step
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

# Execute training by looping over training epochs
EPOCHS = 5
for epoch in range(EPOCHS):
  
  # Reset the metrics at the start of the next epoch
  train_loss.reset_state()
  train_accuracy.reset_state()
  test_loss.reset_state()
  test_accuracy.reset_state()

# Perform batch training for all the available (input, output) pairs
  for images, labels in train_ds:
    train_step(images, labels)

# Perform batch validation for all the available (input, output) pairs
  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result():0.2f}, '
    f'Accuracy: {train_accuracy.result() * 100:0.2f}, '
    f'Test Loss: {test_loss.result():0.2f}, '
    f'Test Accuracy: {test_accuracy.result() * 100:0.2f}'
  )
