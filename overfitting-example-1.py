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