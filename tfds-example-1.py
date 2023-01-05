# This file contains an example how to train a model with Tensorflow Datasets
# and with keras integrated DataSets.
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ml_utils import StopOnLoss


def create_and_train_model(training_images, training_labels, test_images, test_labels, model_prefix):
    model_name = f"{model_prefix}-tfd-example-1-model.h5"
    if os.path.exists(model_name):
        return tf.keras.models.load_model(model_name)

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                                        tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=15, callbacks=[StopOnLoss(0.05)])
    model.evaluate(test_images, test_labels)

    model.save(model_name)
    return model

if __name__ == '__main__':
    # as_supervised=True: returns tuple (input, label) instead of dict
    (tfds_train_images, tfds_train_labels), (tfds_test_images, tfds_test_labels) = tfds.as_numpy(tfds.load('fashion_mnist', split=['train', 'test'], batch_size=-1, as_supervised=True))

    print(tfds_train_images[0].shape) # The shape of the image is (28, 28, 1) which means that the image is 28x28 pixels and has one color channel (grayscale)
    print(tfds_train_labels[0]) # The number printed here is the label of the image (e.g. 2 for a pullover)

    mnist = tf.keras.datasets.mnist
    (keras_train_images, keras_train_labels), (keras_test_images, keras_test_labels) = mnist.load_data()

    print(tfds_train_images.shape) # The shape of the image is (10000, 28, 28, 1) which means that the image is 28x28 pixels and has one color channel (grayscale)
    print(keras_train_images.shape) # The shape of the image is (60000, 28, 28) which means that the image is 28x28 pixels but has no color channel (grayscale)
    keras_train_images = np.expand_dims(keras_train_images, axis=-1) # Now the shape of the image is (60000, 28, 28, 1) and thus the same as the tfds_train_images
    print(keras_train_images.shape)
    keras_test_images = np.expand_dims(keras_test_images, axis=-1)[:10000]

    # Both datasets have the same shape and thus can be used in the same model
    print(keras_test_images.shape)
    print(tfds_test_images.shape)

    # Create and train the model with the keras_train_images
    keras_train_images = keras_train_images / 255.0
    keras_test_images = keras_test_images / 255.0
    model = create_and_train_model(keras_train_images, keras_train_labels, keras_test_images, keras_test_labels, "keras")
    model.summary()
    test_image = keras_test_images[0].reshape(1, 28, 28, 1) # The model expects a batch of images, thus we need to add a dimension
    test_image_classification = model.predict(test_image)
    test_image_classification = np.argmax(test_image_classification[0], axis=0) # The model returns a probability for each class, thus we need to get the class / index with the highest probability
    print(f"The model predicted the image to be a {test_image_classification} and the label of the same image is {keras_test_labels[0]}")

    # Create and train the model with the tfds_train_images
    tfds_train_images = tfds_train_images / 255.0
    tfds_test_images = tfds_test_images / 255.0
    model = create_and_train_model(tfds_train_images, tfds_train_labels, tfds_test_images, tfds_test_labels, "tfds")
    test_image = tfds_test_images[0].reshape(1, 28, 28, 1)
    test_image_classification = model.predict(test_image)
    test_image_classification = np.argmax(test_image_classification[0], axis=0)
    print(f"The model predicted the image to be a {test_image_classification} and the label of the same image is {tfds_test_labels[0]}")

