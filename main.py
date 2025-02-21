"""
Main python file
"""


import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
            tf.keras.layers.Dense(tf.math.reduce_prod(data_shape).numpy(), activation='sigmoid'),
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

    input_shape = x_test.shape[1:]
    dimensions = 64

    autoencoder = SimpleAutoencoder(dimensions, input_shape)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    autoencoder.fit(x_train, x_train,
                    epochs=1,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    encoded_img = autoencoder.encoder(x_test).numpy()
    decoded_img = autoencoder.decoder(encoded_img).numpy()

    count = 6
    plt.figure(figsize=(8, 4))
    for i in range(count):
        ax = plt.subplot(2, count, i + 1)
        plt.imshow(x_test[i])
        plt.title("orig")
        plt.gray()

        ax = plt.subplot(2, count, i + 1 + count)
        plt.imshow(decoded_img[i])
        plt.title("recon")
        plt.gray()
    plt.show()


if __name__ == '__main__':
    main()
