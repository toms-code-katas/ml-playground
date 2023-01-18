import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_model():
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model

current_folder = os.path.dirname(os.path.realpath(__file__))
print(f"Reading data from {current_folder}")


TRAINING_DIR = "tmp/rps-train/rps"
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "tmp/rps-test/rps-test-set"
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=16
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=16
)

model = create_model()

if os.path.exists("rps-2-model.h5"):
    model = tf.keras.models.load_model("rps-2-model.h5")
else:
    model.fit(train_generator, epochs=2, validation_data=validation_generator, verbose=1)
    model.save("rps-2-model.h5")

# Retrieve a batch of images from the test set and plot them
i = 0
for _ in range(16):
    ax = plt.subplot(4, 4, i + 1)
    image_batch, label_batch = validation_generator.next()
    # Take the first image from the batch
    img = image_batch[0]
    # In order to predict the class of the image we need to add a batch dimension
    predict_img = (np.expand_dims(img, 0))
    prediction = model.predict(predict_img)
    plt.imshow(img)
    # prediction has the shape (1, 3) because we added a batch dimension. We need to remove it
    # to get the shape (3,) which are the probabilities for each class
    prediction = np.squeeze(prediction)
    plt.title(np.around(prediction))
    plt.axis("off")
    i = i + 1

plt.show()
