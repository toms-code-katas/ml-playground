import numpy as np
import tensorflow as tf

from ml_utils import StopOnLoss

def secret_function(x):
    y = 2 * x
    return y

def predict(x, model):
    print(f"{model.predict([x], verbose=0)} == {secret_function(x)}")

xs = np.arange(-500, 500, 1, dtype=np.float32)

f = np.vectorize(secret_function)
ys = f(xs)

model_mse = tf.keras.Sequential([
    tf.keras.layers.Dense(name="1_unit", units=1, input_shape=[1]),
    tf.keras.layers.Dense(name="2_unit", units=2),
    tf.keras.layers.Dense(name="3_unit", units=3),
    # tf.keras.layers.Dense(name="output_layer", units=1) # This is the output layer with one unit that predicts the output. Since the output is a scalar, the output layer has only one unit.
    tf.keras.layers.Dense(name="output_layer", units=1) # This output layer has 2 units. The output is a vector with 2 elements.
])
model_mse.compile(optimizer='adam', loss='mean_squared_error')
model_mse.summary()

for layer in model_mse.layers:
    print(layer.name, layer.input_shape, layer.output_shape)

print(model_mse.output_shape)

model_mse.fit(xs, ys, epochs=500, callbacks=[StopOnLoss(0.001)], shuffle=True, verbose=0)

for number in range(-5, 5):
    # If the output layer has only one unit, the output is a scalar
    # For example, the output of the network is [[-10.008559]] for the input -5
    # If the output layer has more than one unit, the output is a vector. For 2 units,
    # the output is [[-10.008559, -10.008559]] for the input -5
    predict(number, model_mse)
