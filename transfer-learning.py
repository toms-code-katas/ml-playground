import os
import random
import requests
import zipfile

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download the weights from https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
# and save them in the same directory as this script
if not os.path.exists("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    r = requests.get(url, allow_redirects=True)
    open("inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", "wb").write(r.content)

# Create the base model from the pre-trained model Inception V3
local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Setting include_top to False means that the last layer of the model is not included
pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)

# print the model summary
pre_trained_model.summary()

# print only the input layer of the model
print(pre_trained_model.input)

# The input shape of the model is (150, 150, 3) which means that the model expects an image of size 150x150 with 3 channels (RGB)
# In order to change the input shape of the model, we have to create a new model with the new input shape

x = layers.Input(shape=(250, 250, 3))
y = pre_trained_model(x)

# Print the summary of the new model
model = Model(x, y)
model.summary()

# Print the input layer of the new model
# The shape of the input layer is now (250, 250, 3)
print(model.input)

# Reset the input layer of the model to the new input layer
x = layers.Input(shape=(150, 150, 3))
y = pre_trained_model(x)
model = Model(x, y)

# Set trainable to False to freeze the layers
for layer in model.layers:
    layer.trainable = False

# Add a new output layers with 2 classes

# The flatten layer is used to flatten the output of the last layer of the model
# which is a 5x5x2048 tensor to a 1D tensor with 51200 elements.
# The 51200 elements are then passed to the dense layer with 1024 neurons.
# The Dense layer takes the input and multiplies it with the weights and adds the bias.
x = layers.Flatten()(model.output)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation="sigmoid")(x)

# Compile the model
model = Model(model.input, x)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Now lets train the model with the cats and dogs dataset
# Download the dataset from https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip
# and extract it in the same directory as this script

if not os.path.exists("cats_and_dogs_filtered.zip"):
    url = "https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip"
    r = requests.get(url, allow_redirects=True)
    open("cats_and_dogs_filtered.zip", "wb").write(r.content)
    with zipfile.ZipFile("cats_and_dogs_filtered.zip", "r") as zip_ref:
        zip_ref.extractall(".")

# Now create the training and validation data generators
train_dir = "cats_and_dogs_filtered/train"
validation_dir = "cats_and_dogs_filtered/validation"


# Load the whole model if it has already been saved
if os.path.exists("cats_and_dogs.h5"):
    model = tf.keras.models.load_model('cats_and_dogs.h5')
else:
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=20,
                                                                  class_mode="binary",
                                                                  target_size=(150, 150))
    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10, verbose=1)
    model.save("cats_and_dogs.h5")
    # Save the history as well using pickle
    import pickle
    with open("history.pickle", "wb") as f:
        pickle.dump(history.history, f)

# Predict a single image
import numpy as np
from tensorflow.keras.preprocessing import image

# Select a random image from the validation set
validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")


for i in range(10):
    validation_cats = os.listdir(validation_cats_dir)
    validation_dogs = os.listdir(validation_dogs_dir)
    random_cat = random.choice(validation_cats)
    random_dog = random.choice(validation_dogs)
    random_cat_path = os.path.join(validation_cats_dir, random_cat)
    random_dog_path = os.path.join(validation_dogs_dir, random_dog)

    img = image.load_img(random_cat_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print("dog")
    else:
        print("cat")

    # Now display the image for verification
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

