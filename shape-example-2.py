import numpy
import tensorflow as tf
import numpy as np

fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

print(help(fmnist.load_data)) # Prints the docstring of the function. In this case it includes the labels and their meaning.

print(training_images.shape) # (60000, 28, 28) means 60000 images of 28x28 pixels

# Set the print options so that we can see one line of the array in one line of the console
np.set_printoptions(linewidth=320)

index = 0 # Should be a boot

# Print the label and the corresponding image
print(f'Label: {training_labels[index]}')
print(f'Original image: {training_images[index]}')
print(f'Image normalized: {np.around(training_images[index]/255, 2)}')
# The normalized image contains values which are much closer together than the original image.

training_images = training_images / 255.0 # Normalize the training images to values between 0 and 1
test_images = test_images /255.0
# Normalization in this case means that the values are scaled down to values between 0 and 1.
# This is done because the original values are between 0 and 255, which is a very large range.
# Using smaller values makes similar values closer together, which makes it easier for the model to learn.
# See here: https://medium.com/analytics-vidhya/a-tip-a-day-python-tip-8-why-should-we-normalize-image-pixel-values-or-divide-by-255-4608ac5cd26a


# The flatten layer is needed because the input data is 2D (28x28) and the Dense layer expects 1D data
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=2, validation_data=(test_images, test_labels), verbose=1)

model.summary()

# The output shape of the flatten layer is (None, 784) where None is the number of images in the input data
# and 784 is the number of pixels in the image (28x28), Thus the 2d array of the image is flattened to a 1d array
# The output shape of the first dense layer is (None, 128) where None is the number of images in the input data
# and 128 is the number of neurons in the layer.
# The output shape of the second dense layer is (None, 10) where None is the number of images in the input data
# and 10 is the number of neurons in the layer.
# Softmax is a function that takes as input a vector of K real numbers, and normalizes it
# into a probability distribution consisting of K probabilities
# Thus the output of the softmax layer is a vector of 10 probabilities that sum to 1.0
for layer in model.layers:
    print(f"Layer {layer.name}, input_shape={layer.input_shape}, output_shape={layer.output_shape}")

test_image = (test_images[0] * 255).astype(np.int32) # Convert image back to 8 bit integer values
test_image_label = test_labels[0]

print(f'Test image label: {test_image_label}')
print(f'Test Image: {test_image}')

predicted_label = model.predict(np.array([test_image])) # Needs to be an array because the model expects a 4D array with shape (None, 28, 28, 1)
predicted_label = numpy.argmax(predicted_label) # Get the index of the highest probability

print(f'Predicted label: is {predicted_label}. Actual label is {test_image_label}.')
