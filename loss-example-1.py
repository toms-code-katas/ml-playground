import tensorflow as tf

print (tf.keras.losses.mean_absolute_error(tf.ones((2, 2,)), tf.zeros((2, 2))))

# Create a tensor with 1 row containing the value three
all_threes = tf.constant([3], dtype=tf.float32)
# Create a tensor with 1 row containing the value 1
all_ones = tf.constant([1], dtype=tf.float32)

# mean_absolute_error calculates the absolute difference between the two tensors
# that is 3 - 1 = 2
loss = tf.keras.losses.mean_absolute_error(all_threes, all_ones)
print(f"Loss: {loss.numpy()}") # Loss: 2.0

# Try with 2.5 instead of 3 and add an additional value to the tensors
all_2dot5 = tf.constant([2.5, 2.5], dtype=tf.float32)

# In this case the loss is 2.5 - 1 = 1.5 for each value thus the mean absolute error is 1.5
loss = tf.keras.losses.mean_absolute_error(all_2dot5, all_ones)
print(f"Loss: {loss.numpy()}") # Loss: 1.5

# Try with different values in the tensors
mixed_values = tf.constant([2, 3], dtype=tf.float32)
# In this case the loss is 2 - 1 = 1 and 3 - 1 = 2 thus the mean absolute error is 3 / 2 = 1.5
loss = tf.keras.losses.mean_absolute_error(mixed_values, all_ones)
print(f"Loss: {loss.numpy()}") # Loss: 1.5