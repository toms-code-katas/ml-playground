import numpy as np
import tensorflow as tf
# Understand Mean Absolute Error (MAE) and how it can be used to evaluate accuracy of
# sequence models.

# Calculate the MAE for a sequence of numbers
# 3 is the mean number in the sequence
numbers = [1, 2, 3, 4, 5]
print(f"MAE: {np.mean(np.abs(numbers))}")

# This can also be calculated using numpy:
print(f"MAE: {np.sum(numbers) / len(numbers)}")

# Now calculate the squared mean absolute error (SMAE):
print(f"SMAE: {np.mean(np.square(numbers))}")

# This can also be calculated using numpy:
print(f"SMAE: {np.sum(np.square(numbers)) / len(numbers)}")

# SMAE is a better metric for evaluating sequence models because it penalizes larger errors

# Multiply the numbers by 2 using numpy:
diff = np.array(numbers) * 2

# Let's use the functions from keras:
print(tf.keras.metrics.mean_absolute_error(numbers, diff).numpy())
print(tf.keras.metrics.mean_squared_error(numbers, diff).numpy())

# The values are the same because the difference is double the original values
