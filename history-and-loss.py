# This file contains an example how to plot the loss of a model
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ml_utils import StopOnLoss


def secret_function(x):
    arbitrary_summand = 0
    y = 2 * x + x / 8 + arbitrary_summand
    return y

def predict(x, model, bias, weight):
    print(f"{model.predict([x], verbose=0)[0][0]} == {secret_function(x)} == {weight * x + bias}")

model = None
history = None

if os.path.exists("history-and-loss-model.h5"):
    model = tf.keras.models.load_model("history-and-loss-model.h5")
else:
    xs = np.arange(-5000, 5000, 1, dtype=np.float32)

    # Vectorize the function to apply it to all elements of the array
    f = np.vectorize(secret_function)
    ys = f(xs)

    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1], activation="linear")])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    history = model.fit(xs, ys, epochs=500, callbacks=[StopOnLoss(0.0001)], shuffle=True, verbose=1,
                        validation_split=0.2)
    model.save("history-and-loss-model.h5")

if os.path.exists("history-and-loss-history.pkl"):
    with open("history-and-loss-history.pkl", "rb") as f:
        history = pickle.load(f)
else:
    with open("history-and-loss-history.pkl", "wb") as f:
        pickle.dump(history, f)

# Plot the loos for all epochs
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()

# Plot the loss for the last x epochs
last_x_epoch = 18
loss_last_x_epochs = loss[-last_x_epoch:]
val_loss_last_x_epochs = val_loss[-last_x_epoch:]

epochs = range(0, last_x_epoch, 1)

plt.plot(epochs, loss_last_x_epochs, 'r', label='Training loss')
plt.plot(epochs, val_loss_last_x_epochs, 'b', label='Validation loss')
plt.title(f'Training and validation loss (last {last_x_epoch} epochs)')
plt.legend(loc=0)
plt.figure()
plt.show()
