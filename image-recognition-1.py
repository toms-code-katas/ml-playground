# This file contains an example of image recognition using convolutional neural networks.
# A good visualization of what convolutions are can be found here:
# https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif
import os
import tensorflow as tf
import matplotlib.pyplot as plt

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
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([

    # The first convolutional layer takes the input image and applies 64 filters to it.
    # The filters are 3x3 matrices that are applied to the image.
    # Note that the output shape of the convolutional layer is (None, 26, 26, 32)
    # where None is the number of images in the input data, 26 is the number of pixels in the image
    # minus 2 because the filter is 3x3 and 32 is the number of filters applied to the image.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Use same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if os.path.isfile('model.h5'):
    model.load_weights('model.h5')
else:
    model.fit(training_images, training_labels, epochs=5, validation_data=(test_images, test_labels), verbose=1)
    model.save_weights('model.h5')

# Now visualize the output of the first convolutional layer for the first image in the training set
image = test_images[0]
image = image.reshape(1, 28, 28, 1) # Needs to be reshaped to (1, 28, 28, 1) because the model expects a batch of images
conv2d_layer1 = model.layers[0]
conv2d_image = conv2d_layer1(image) # Run the image through the first convolutional layer. This will produce 32 images as a 4d array (1, 26, 26, 32)
print_image_summary(conv2d_image)