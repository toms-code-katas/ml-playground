import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# Overfitting is a common problem in machine learning. It occurs when a model
# performs well on the training data, but does not generalize well to new data.
# Indicators of overfitting include:
# * The training accuracy is much greater than the validation accuracy.
# * The validation loss is much greater than the training loss.
# * The training loss is decreasing, but the validation loss is increasing.
# Overfitting can ber reduced by:
# * Adjusting the training rate of the optimizer. Because the model then learns more slowly.
# * In NLP, reducing the size of the vocabulary, if the frequency of words is not distributed evenly (e.g. 80% of the words are used only once).
# * In NLP, reducing the dimensionality of the embedding layer.
# * Adding dropout layers. This should be done after the vocabulary size and embedding layer dimensionality have been reduced.
# * Adding regularization. Regularization reduces the polarity of the weights, so that the model does not rely on a few weights to make predictions.
#
# See chapter6/emotion_classifier_binary.ipynb for regularization and chapter 6 as well as whole for more details.
# See chapter7/sarcasm_simplernn.ipynb for a learning rate example.

# The Dropout layer randomly sets input units to 0 so that their contribution to the neurons in the next layer is reduced.
# This helps prevents the neurons from co-adapting too much.
# The following example shows how the neurons are randomly set to 0.
# Note that the model learns slower, but predictions are more accurate.

# The kernel_regularizer parameter is used to add regularization to the model. In this example
# l2 regularization adds a penalty to the loss function for large weights. The higher the
# regularization parameter, the more the weights are penalized.
# For NLP most common regularization is l2

model = tf.keras.Sequential([
    keras.layers.Dense(units=2, input_shape=[1], kernel_regularizer=keras.regularizers.l2(0.1)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500, verbose=0)

print(model.predict([10.0]))


