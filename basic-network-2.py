# This file contains an example of a simple neural network
# that can be used to predict the output of a function
# that is not known to the network.
# The network is trained to predict the output of the function
# secret_function(x) = 2 * x + x / 8
import numpy as np
import tensorflow as tf

from ml_utils import StopOnLoss

def secret_function(x):
    """This function does some calculations and returns a result. It is used to generate the training and validation data."""
    # This arbitrary summand can be used to make the training harder
    # by obfuscating the relationship between the input and the output
    arbitrary_summand = 0
    y = 2 * x + x / 8 + arbitrary_summand
    return y

def predict(x, model, bias, weight):
    print(f"{model.predict([x], verbose=0)[0][0]} == {secret_function(x)} == {weight * x + bias}")

xs = np.arange(-5000, 5000, 1, dtype=np.float32)

# Vectorize the function to apply it to all elements of the array
f = np.vectorize(secret_function)
ys = f(xs)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.fit(xs, ys, epochs=500, callbacks=[StopOnLoss(0.0001)], shuffle=True, verbose=0)

# Get the bias and weight from the model
# The bias and weight are the two parameters of the model and are learned during training
bias = model.layers[0].bias.numpy()[0]
weight = model.layers[0].weights[0].numpy()[0][0]

for number in range(-5, 5):
    predict(number, model, bias, weight)
