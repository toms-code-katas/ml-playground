import io
import os
import string
import tempfile
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
             "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
             "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his",
             "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell",
             "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them",
             "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this",
             "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were",
             "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos",
             "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours",
             "yourself",
             "yourselves"]

# This table is used to remove punctuation from a string. From the documentation:
# If a third argument is given, it must be a string, whose characters will be
# deleted from the input string.
punctuation_translation_table = str.maketrans('', '', string.punctuation)

class StopOnLoss(tf.keras.callbacks.Callback):

    def __init__(self, loss=1.0e-05):
        self.__loss = loss

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('loss') <= self.__loss:
            print(f"Reached {self.__loss} loss")
            self.model.stop_training = True

def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

def pre_process_text(text):
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace("-", " - ")
    text = text.replace("/", " / ")
    text = text.replace("’", "")

    # Remove html tags
    soup = BeautifulSoup(text, features="html.parser")
    text = soup.get_text()

    # Next remove punctuation
    text = text.translate(punctuation_translation_table)

    # Now remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def print_vocabulary_of_tokenizer(tokenizer, items=10):
    # Get the first 10 items of the vocabulary
    vocabulary = list(tokenizer.word_index.items())[:items]
    for word, index in vocabulary:
        print(f"{word} -> {index}")

def print_decoded_sequence(sequence, tokenizer):
    for index in sequence:
        print(f"{index} -> {tokenizer.index_word[index]}")


def visualize_embeddings_from_model(model, embedding_layer_name, tokenizer):
    """
    This function will create a file that can be used by the embedding projector
    See https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/ungraded_labs/C3_W2_Lab_1_imdb.ipynb
    """
    # Get the embedding layer from the model
    embedding_layer = model.get_layer(embedding_layer_name)

    # Get the weights from the embedding layer
    weights = embedding_layer.get_weights()[0]

    # Get the vocabulary from the tokenizer
    vocab = tokenizer.word_index
    reverse_word_index = dict([(value, key) for (key, value) in vocab.items()])

    # The vocabulary size is set to the first item in the weights shape
    vocab_size = weights.shape[0]
    print(f"Vocabulary size: {vocab_size}")

    # Create a temporary directory for the files
    temp_dir = tempfile.mkdtemp()

    # Create the files for the embedding vectors and the words in the temporary directory
    out_v = io.open(os.path.join(temp_dir, 'vecs.tsv'), 'w', encoding='utf-8')
    out_m = io.open(os.path.join(temp_dir, 'meta.tsv'), 'w', encoding='utf-8')

    # Write the embedding vectors and the words to the files
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    # Close the files
    out_v.close()
    out_m.close()

    return out_v, out_m

def plot_word_count_distribution(tokenizer, from_index=0, to_index=100, print_words=True):
    wc = tokenizer.word_counts
    # Sort the word counts by their count
    wc = dict(sorted(wc.items(), key=lambda item: item[1], reverse=True))
    # Get the range of words to plot
    wc = dict(list(wc.items())[from_index:to_index])

    x_counter = []
    word_count = []
    curr_x = 1
    for item in wc:
        x_counter.append(curr_x)
        curr_x = curr_x + 1
        word_count.append(wc[item])

    plt.plot(x_counter, word_count)
    plt.show()

    # Display the words and their counts
    if print_words:
        for item in wc:
            print(f"{item}: {wc[item]}")


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def trend(time, slope=0):
    series = slope * time
    return series


def seasonal_pattern(season_time):
    data_pattern = np.where(season_time < 0.4,
                            np.cos(season_time * 2 * np.pi),
                            1 / np.exp(3 * season_time))
    return data_pattern


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    data_pattern = amplitude * seasonal_pattern(season_time)
    return data_pattern


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise


def plot_series(x, y, format="-", start=0, end=None,
                title=None, xlabel=None, ylabel=None, legend=None):
    plt.figure(figsize=(10, 6))

    # Check if there are more than two series to plot
    if type(y) is tuple:

        # Loop over the y elements
        for y_curr in y:
            # Plot the x and current y values
            plt.plot(x[start:end], y_curr[start:end], format)

    else:
        # Plot the x and y values
        plt.plot(x[start:end], y[start:end], format)

    # Label the x-axis
    plt.xlabel(xlabel)

    # Label the y-axis
    plt.ylabel(ylabel)

    # Set the legend
    if legend:
        plt.legend(legend)

    # Set the title
    plt.title(title)

    # Overlay a grid on the graph
    plt.grid(True)