import numpy as np
# What is the difference between a univariate and multivariate time series?
# Univariate time series are time series with only one variable, e.g. the temperature at a certain location.
# Multivariate time series are time series with more than one variable, e.g. the temperature at a certain location and the humidity at the same location.
# Here is an example of a univariate time series:
univariate_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Here is an example of a multivariate time series:
multivariate_series = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

# See https://stats.stackexchange.com/questions/342754/what-is-the-difference-between-univariate-and-multivariate-time-series