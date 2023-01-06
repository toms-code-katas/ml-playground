import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    test_data = np.random.randint(10, 90, (1000, 25, 25, 1)) # 1000 images of 25x25 pixels for example
    print(test_data.shape[1:]) # (25, 25, 1), the shape of the image data (without the number of images)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=test_data.shape[1:])) # input_shape is the shape of the image data (without the number of images), see above
    model.compile(optimizer='adam', loss='mean_squared_error')

    # The model expects a 4D array with shape (None, 25, 25, 1)
    # where None is the number of rows in the input of the test data
    # and 25, 25, 1 are the dimensions of the image data (without the number of images)
    model.summary()
    print(model.layers[0].output_shape) # The output shape of the first layer is the same as the input shape because it is a Dense layer with 1 neuron

    # history = model.fit(test_data, test_data, epochs=100, validation_split=0.2, shuffle=True, verbose=1)
