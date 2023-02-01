# This file contains an example of image recognition using convolutional neural networks.
# A good visualization of what convolutions are can be found here:
# https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif
import glob
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

# This functions prints the label based on the prediction of the model
def print_label(prediction):
    # First define the labels and the corresponding indices
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Now create a dictionary with the labels and the indices
    label_dict = dict(zip(indices, labels))
    # Now get the maximum value from the prediction
    max_value = max(prediction)
    # Now get the index of the maximum value
    max_index = list(prediction).index(max_value)
    # Now print the label
    print(f"Label: {label_dict[max_index]}")

def run_image_through_layers(test_image):
    # Print the image
    plt.imshow(test_image, cmap='inferno')
    plt.show()
    test_image = test_image.reshape(1, 28, 28,
                                    1)  # Needs to be reshaped to (1, 28, 28, 1) because the model expects a batch of images
    conv2d_layer1 = model.layers[0]
    conv2d_image = conv2d_layer1(
        test_image)  # Run the image through the first convolutional layer. This will produce 32 images as a 4d array (1, 26, 26, 32)
    print_image_summary(conv2d_image)
    # Run the image through the first max pooling layer
    max_pooling_layer1 = model.layers[1]
    conv2d_image = max_pooling_layer1(
        conv2d_image)  # Run the image through the first max pooling layer. This will produce 32 images as a 4d array (1, 13, 13, 32)
    print_image_summary(conv2d_image)
    # Now run the image through the second convolutional layer
    conv2d_layer2 = model.layers[2]
    conv2d_image = conv2d_layer2(
        conv2d_image)  # Run the image through the second convolutional layer. This will produce 32 images as a 4d array (1, 24, 24, 32)
    print_image_summary(conv2d_image)
    # Run the image through the second max pooling layer
    max_pooling_layer2 = model.layers[3]
    conv2d_image = max_pooling_layer2(
        conv2d_image)  # Run the image through the second max pooling layer. This will produce 32 images as a 4d array (1, 12, 12, 32)
    print_image_summary(conv2d_image)
    # Run the image through the flatten layer
    flatten_layer = model.layers[4]
    conv2d_image = flatten_layer(conv2d_image)  # Run the image through the flatten layer.
    print(
        f"conv2d_image.shape={conv2d_image.shape}")  # The shape is (1, 800) because the image is flattened to a 1d array which is 12*12*32=800 (see above)
    # Run the image through the first dense layer
    dense_layer1 = model.layers[5]
    conv2d_image = dense_layer1(conv2d_image)  # Run the image through the first dense layer.
    print(
        f"conv2d_image.shape={conv2d_image.shape}")  # The shape is (1, 128) because the layer has 128 neurons and therefore produces a 1d array of 128 values
    # Let's display the output of the first dense layer which is a 1d array of 128 values
    plt.figure(figsize=(10, 4))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(conv2d_image, cmap='inferno')
    plt.show()
    # The image above shows the output of the first dense layer which is a 1d array of 128 values. The black pixels represent the values that are close to 0 and the white pixels represent the values that are close to 1.
    # Run the image through the output layer
    dense_layer2 = model.layers[6]
    conv2d_image = dense_layer2(conv2d_image)  # Run the image through the output layer.
    print(
        f"conv2d_image.shape={conv2d_image.shape}")  # The shape is (1, 10) because the layer has 10 neurons and therefore produces a 1d array of 10 values
    # Print the array surpessing the scientific notation and rounding the values to 2 decimals
    np.set_printoptions(suppress=True, precision=2)
    print(conv2d_image)
    print_label(conv2d_image[0])


def print_image_summary(conv2d_image, cols=8):
    """  Prints a summary of the image. See https://www.kaggle.com/code/sanjitschouhan/visualizing-conv2d-output?scriptVersionId=49603115&cellId=27
    """
    print(f"conv2d_image.shape={conv2d_image.shape}") # (1, 26, 26, 32) because there is only one image and 32 filters
    channels = conv2d_image.shape[-1] # 32 from (1, 26, 26, 32)
    images = conv2d_image[0] # (26, 26, 32) All the images produced by the 32 filters
    rows = channels // cols # Just for display purposes
    plt.figure(figsize=(cols*2,rows*2))
    for i in range(channels):       # Iterate over the 32 filters
        plt.subplot(rows,cols,i+1)  # Create a subplot for each filter
        plt.xticks([])       # Remove the x-axis ticks
        plt.yticks([])       # Remove the y-axis ticks
        plt.imshow(images[:,:,i], cmap='inferno') # [:,:,i] Since this is a 3d array, this means all the rows and columns of the i-th filter
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (real_test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
real_test_images = real_test_images / 255.0

model = tf.keras.models.Sequential([

    # The first convolutional layer takes the input image and applies 64 filters to it.
    # The filters are 3x3 matrices that are applied to the image.
    # Note that the output shape of the convolutional layer is (None, 26, 26, 32)
    # where None is the number of images in the input data, 26 is the number of pixels in the image
    # minus 2 because the filter is 3x3 and 32 is the number of filters applied to the image.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # The max pooling layer reduces the size of the image by half
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Need to flatten the output of the convolutional layers to feed it to the dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # The output layer has 10 neurons, one for each class
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Use same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if os.path.isfile('model.h5'):
    model.load_weights('model.h5')
else:
    model.fit(training_images, training_labels, epochs=5, validation_data=(real_test_images, test_labels), verbose=1)
    model.save_weights('model.h5')

# The following is used to visualize the way an image is processed by the model
# Visualize the output of the first convolutional layer for the first image in the training set
test_image = real_test_images[0]

run_image_through_layers(test_image)

# Now try to predict the class of a range of images located in the image-recognition-test-images folder.
# As a reminder, the classes are:
# 0 - T-shirt/top
# 1 - Trouser
# 2 - Pullover
# 3 - Dress
# 4 - Coat
# 5 - Sandal
# 6 - Shirt
# 7 - Sneaker
# 8 - Bag
# 9 - Ankle boot

# Get a list of all the images in the image-recognition-test-images folder
real_test_images = glob.glob("image-recognition-test-images/*.png")
for real_test_image in real_test_images:
    # Since it is an arbitrary image, we need to convert it to a 28x28 image with a single channel
    img = image.load_img(real_test_image)
    plt.imshow(img)
    plt.show()

    # For the image to be processed by the model, it needs to be converted to a 28x28 in grayscale
    img = img.resize((28, 28))
    img = img.convert('L')
    # Plot the image in grayscale
    plt.imshow(img, cmap='gray')
    plt.show()

    # For running the image through the model, we need to convert it to a numpy array
    img = image.img_to_array(img)
    run_image_through_layers(img)
    #
    # # Convert the image to a numpy array and predict the class
    # img = image.img_to_array(img)
    # img = img.reshape(1, 28, 28, 1)
    # img = img / 255.0
    # prediction = model.predict(img)
    # print(f"prediction={prediction}")
