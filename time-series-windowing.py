import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import ml_utils
# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
# Window = Series of values to predict the next value for, learn on respectively
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

# Print the result
for window_dataset in dataset:
  print([item.numpy() for item in window_dataset])

# Prepare data for time series learning.
# Flatten, batching and mapping. See here https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_1_features_and_labels.ipynb

# Do the same with a plain python list
print("")
simple_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ds = tf.data.Dataset.from_tensor_slices(simple_series)
ds = ds.window(size=5, shift=1, drop_remainder=True)
for window_dataset in ds:
  print([item.numpy() for item in window_dataset])


# Generate some synthetic data as done in the tf book:
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + ml_utils.trend(time, slope) + ml_utils.seasonality(time, period=365, amplitude=amplitude)

# Update with noise
series += ml_utils.noise(time, noise_level, seed=42)

# Plot the results
# The series is a numpy array with shape (1461,), so it is a simple 1D array
ml_utils.plot_series(time, series, xlabel='Time', ylabel='Value')

# No try to print the simple_series
time = np.arange(len(simple_series), dtype="float32")
# The plotted graph is a simple line from 0,0 to 9,9
ml_utils.plot_series(time, simple_series, xlabel='Time', ylabel='Value')

plt.show()