# Script created by copy-pasting code to help Davide Martire, by PeterC 29-04-2024
# Import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Layer
from keras.initializers import RandomUniform, Constant
from tensorflow import math
from keras.initializers import Initializer
from sklearn.cluster import KMeans

class RBF_Layer2(layers.Layer):
    def _init_(self, units, init_beta=1.0, initializer=None): # Method for instantiation the layer object
        super(RBF_Layer2, self)._init_()
        self.units = units
        self.init_beta = init_beta
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer

    def build(self, input_shape): # Method to initialize weights --> called automatically at instantiation
        super(RBF_Layer2,self).build(input_shape)
        self.centers = self.add_weight(
            name='centers',
            shape=(input_shape[-1], self.units),
            initializer=self.initializer,
            trainable=False,
        )
        self.betas = self.add_weight(
            name='betas',
            shape=(self.units,),
            initializer=Constant(value=self.init_beta),
            trainable=True,
        )

    def call(self, inputs): # Method for forward pass (i.e. evaluation)?

        H = tf.transpose(self.centers - inputs)
        norm2 = -self.init_beta*tf.norm(H, ord='euclidean', axis=0)**2
        return tf.expand_dims(math.exp(norm2), axis=0)

    def compute_output_shape(self, input_shape): # Method to get shape of the output vector from the layer?
        return input_shape[-1], self.units

class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def _init_(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def _call_(self, shape, dtype=None):
        assert shape[0] == self.X.shape[1]

        n_centers = shape[1]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_
    

if __name__=="__main__":
    # Define training and validation sets
    X_train, X_test = tf.split(data, [int(0.8 * len(data)), int(0.2 * len(data))])
    y_train, y_test = tf.split(rhos2, [int(0.8 * len(label)), int(0.2 * len(label ))])

    # Define RBF network model
    # DEVNOTE: issue seems to be in the interfaces between the custom layer and the other layers: TF returns error when evaluating the kernel

    model = tf.keras.Sequential([
    keras.Input(shape = (16,)), # Input size is 16, while hidden layer has 15?
    RBF_Layer2(15, 3, InitCentersKMeans(X_train)), # KMeans to generate centres of the Radial basis functions from data?
    keras.layers.Dense(1)

    ])

    model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

    model.fit(tf.expand_dims(X_train, axis=-1), y_train, batch_size=10, epochs=2, verbose=2)