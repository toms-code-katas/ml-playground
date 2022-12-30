import numpy as np
import tensorflow as tf

import ml_utils
from ml_utils import StopOnLoss

def secret_function(x, y):
    z = x + y
    return z


def generate_test_data():
    xs = np.arange(-5000, 5000, 1, dtype=np.float32)
    ys = np.arange(-5000, 5000, 1, dtype=np.float32)
    zs = secret_function(xs, ys)
    return xs, ys, zs


def create_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2])])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


model = create_model()

xs, ys, zs = generate_test_data()

# xs and ys are 1D arrays, so we need to stack them to create a 2D array
# with shape (10000, 2) where each row is a pair of values
# from xs and ys and is compatible with the input_shape of the model
# which is (None,2) where None is the number of rows in the input
# and 2 is the number of columns in the input which is the number of
# features in the input or in this case the number of parameters of the

xs_ys_in2d = np.vstack((xs, ys)).T
print(f"xs: {xs.shape}, ys: {ys.shape}, zs: {zs.shape}, xyin2d: {xs_ys_in2d.shape} model: {model.input_shape}")

history = model.fit(xs_ys_in2d, zs, epochs=100, callbacks=[StopOnLoss()], validation_split=0.2,
                    shuffle=True, verbose=0)

print(model.predict([[2.0, 3.0]]))
print(secret_function(2.0, 3.0))

print(history.history.keys())
ml_utils.plot_loss(history)
