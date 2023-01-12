import tensorflow as tf

foo = tf.constant([-10, -5, 0.0, 5, 10], dtype=tf.float32)

# the relu function is max(0, x) where x is the input tensor
print(tf.keras.activations.relu(foo).numpy())

# The softmax function is exp(x) / sum(exp(x)) where x is the input tensor
# Note that the softmax function is applied to each row of the input tensor
# so the output tensor will have the same shape as the input tensor
# The softmax function is used in the output layer of a classification model

# Create a 2d tensor
# The second row is all zeros except for the last element which is 100 so the
# softmax function will return [0, 0, 0, 0, 1] because the last element is the largest
# by a large margin
# The first row has values that are closer together so the softmax function will
# return a more even distribution, like [0.01, 0.02, 0.03, 0.04, 0.9]
foo = tf.constant([[-10, -5, 0.0, 5, 10], [0, 0, 0, 0, 100]], dtype=tf.float32)

print(tf.keras.activations.softmax(foo).numpy())

# Now lets test that with a model
model = tf.keras.models.Sequential([tf.keras.layers.Activation('softmax', input_shape=(5,))])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

print(model.predict([[-10, -5, 0.0, 5, 10], [0, 0, 0, 0, 100]]))  # The same as the softmax function
print(model.predict([[1, 2, 3, 4, 5], [100, 0, 0, 0, 0]]))
# The model behaves exactly the same as the softmax function because the model is
# just a wrapper around the softmax function

# What do we need activation functions for?
# An activation function is a deceptively small mathematical expression which decides whether
# a neuron fires up or not. This means that the activation function suppresses the neurons
# whose inputs are of no significance to the overall application of the neural network.
# This is why neural networks require such functions which provide
# significant improvement in performance.
#
# Activation functions are in most cases required to provide some kind of non-linearity
# to the network. Without these functions, the neural network basically becomes a simple
# linear regression model.
# See https://towardsdatascience.com/what-are-activation-functions-in-deep-learning-cc4f01e1cf5c