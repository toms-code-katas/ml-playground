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

# dataset is a tf.data.Dataset object with two elements: (data, labels). Since we split the data into training and validation
# we have two datasets: train_dataset and validation_dataset. The method image_dataset_from_directory in this case
# returns a tuple (train_dataset, validation_dataset)
# Using label_mode='categorical' we get a one-hot encoded vector for the labels
dataset = tf.keras.preprocessing.image_dataset_from_directory(current_folder + "/tmp/rps-train/rps",
                                                              labels="inferred", label_mode="categorical",
                                                              color_mode="rgb", batch_size=16,
                                                              image_size=(150, 150),
                                                              seed=123, shuffle=True, validation_split=0.2,
                                                              subset="both")

model = create_model()

if os.path.exists("rps-model.h5"):
    model = tf.keras.models.load_model("rps-model.h5")
else:
    model.fit(dataset[0], epochs=5, validation_data=dataset[1])
    model.save("rps-model.h5")

for image_batch, label_batch in dataset[1].take(1):

    # Predict the labels of the images in the batch. The predictions are very close to the actual labels
    # because the model was trained on the same data that is used for validation. The model
    # most likely suffers from overfitting
    predictions = model.predict(image_batch)
    print(f"Predictions: {predictions} and labels: {label_batch}")

    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    # The first 16 images in the batch
    # image_batch.shape = (16, 150, 150, 3) that is 16
    for i in range(image_batch.shape[0]):
        # Create a subplot with 4 rows and 4 columns result in 16 images
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(np.around(predictions[i]))
        plt.axis("off")

plt.show()
