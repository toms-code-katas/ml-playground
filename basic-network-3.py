import numpy as np
import tensorflow as tf

from ml_utils import StopOnLoss

def secret_function(x):
    """This function does some calculations and returns a result. It is used to generate the training and validation data."""
    y = 2 * x
    if x == 10000 or x == -10000:
        y = y + 3
    return y

def predict(x, model, bias, weight):
    print(f"{model.predict([x], verbose=0)[0][0]} == {secret_function(x)} == {weight * x + bias}")

xs = np.arange(-10000, 10000, 1, dtype=np.float32)

# Vectorize the function to apply it to all elements of the array
f = np.vectorize(secret_function)
ys = f(xs)

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

model.fit(xs, ys, epochs=500, callbacks=[StopOnLoss(0.001)], shuffle=True, verbose=1)

# Get the bias and weight from the model
# The bias and weight are the two parameters of the model and are learned during training
bias = model.layers[0].bias.numpy()[0]
weight = model.layers[0].weights[0].numpy()[0][0]

for number in range(-5, 5):
    predict(number, model, bias, weight)

# The predicted value is not the same as the secret function value
# because the network is not able to learn the function
# because the function is not linear
# The network is only able to learn linear functions
predict(10000, model, bias, weight)