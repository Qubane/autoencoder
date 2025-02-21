"""
Main python file
"""


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class SimpleAutoencoder(tf.keras.models.Model):
    def __init__(self, dimensions, data_shape):
        super().__init__()

        self.dimensions = dimensions
        self.data_shape = data_shape

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dimensions, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(tf.math.reduce_prod(data_shape), activation='sigmoid'),
            tf.keras.layers.Reshape(data_shape)
        ])

    def call(self, input_data):
        encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data


def main():
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255


if __name__ == '__main__':
    main()
