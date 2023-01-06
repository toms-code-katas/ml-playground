import csv
import json
import os
import string
import urllib.request
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from ml_utils import plot_graphs


# Stopwords are words that are filtered out of the text before processing
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

# This table is used to remove punctuation from the text
# From the documentation: "If there is a third argument, it must be a string,
# whose characters will be mapped to None in the result."
# string.punctuation contains all the punctuation characters, e.g. !"#$%&'()*
translation_table = str.maketrans('', '', string.punctuation)

# The maximum number of words to be used by the tokenizer
vocab_size = 2000

# The dimension of the embedding vector
embedding_dim = 6

# The maximum length of a sentence after padding and truncating
max_length = 100

# The type of truncation to be used, either 'pre' or 'post'.
# Which means that if the sentence is longer than max_length, it will be truncated
# from the beginning or the end of the sentence.
trunc_type='post'

# The type of padding to be used, either 'pre' or 'post'. Which means that if the sentence is
# shorter than max_length, it will be padded from the beginning or the end of the sentence.
padding_type='post'

# The out of vocabulary token
oov_tok = "<OOV>"

# The size of the training set
training_size = 28000


def download_test_data_as_csv():
    """ Download the test data from the internet. """
    if not os.path.exists('binary-emotion.csv'):
        url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/binary-emotion.csv'
        urllib.request.urlretrieve(url, 'binary-emotion.csv')
    return 'binary-emotion.csv'


def download_test_data_as_json():
    """ Download the test data from the internet. """
    if not os.path.exists('binary-emotion.json'):
        url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/binary-emotion.json'
        urllib.request.urlretrieve(url, 'binary-emotion.json')
    return 'binary-emotion.json'


def preprocess_text(text):
    """ Preprocess text by removing html tags, punctuation and stopwords. """
    sentence = text.lower()
    # surround punctuation with spaces so that it can be removed later
    # e.g. "I'm" -> "I ' m" -> "Im"
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    # Remove html tags
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        # Remove punctuation
        word = word.translate(translation_table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    return filtered_sentence.strip()


def get_sentence_and_label_from_csv_file(csv_file_name, max_rows=100000000):
    """ Read the csv file and return the sentences and labels. """
    sentence_list = []
    label_list = []
    with open(csv_file_name, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        rows_read = 0
        for row in reader:
            label_list.append(int(row[0])) # The first row is the label
            sentence_list.append(preprocess_text(row[1])) # The second row is the sentence
            rows_read += 1
            if rows_read >= max_rows:
                break
    return sentence_list, label_list

def get_sentence_and_label_from_json_file(json_file_name, max_rows=100000000):
    """ Read the json file and return the sentences and labels. """
    sentence_list = []
    label_list = []
    with open(json_file_name, 'r') as json_file:
        json_data = json.load(json_file)
        rows_read = 0
        for data in json_data:
            label_list.append(int(data["emotion"]))
            sentence_list.append(preprocess_text(data["sentence"]))
            if rows_read >= max_rows:
                break
    return sentence_list, label_list


def pad_sequences(sequences):
    """ Pad the sequence to the given max_length. """
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return padded_sequences


def get_train_and_validation_sentences_and_labels(emotion_file_name):
    if emotion_file_name.endswith('.json'):
        sentence_list, label_list = get_sentence_and_label_from_json_file(emotion_file_name)
    elif emotion_file_name.endswith('.csv'):
        sentence_list, label_list = get_sentence_and_label_from_csv_file(emotion_file_name)
    train_sentences = sentence_list[0:training_size]
    val_sentences = sentence_list[training_size:]
    train_lbls = label_list[0:training_size]
    val_lbls = label_list[training_size:]
    return train_sentences, val_sentences, train_lbls, val_lbls


def get_train_and_validation_sequences(tokenizer, emotion_file_name):
    """ Get the train and validation sequences. """
    train_sentences, val_sentences, train_lbls, val_lbls = get_train_and_validation_sentences_and_labels(emotion_file_name)
    if tokenizer:
        tokenizer.fit_on_texts(train_sentences)
    train_seqs = tokenizer.texts_to_sequences(train_sentences)
    train_seqs = pad_sequences(train_seqs)
    val_seqs = tokenizer.texts_to_sequences(val_sentences)
    val_seqs = pad_sequences(val_seqs)
    return train_seqs, val_seqs, train_lbls, val_lbls


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.l2(0.025),
                              activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_new_model(text_vectorization_layer):
    model = tf.keras.Sequential([
        text_vectorization_layer,
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.l2(0.025),
                              activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    emotions_file_name = download_test_data_as_json()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    train_seqs, val_seqs, train_labels, val_labels = get_train_and_validation_sequences(tokenizer, emotions_file_name)
    # Needs to be a numpy array because of the way the model is defined
    train_seqs = np.array(train_seqs)
    val_seqs = np.array(val_seqs)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    wc = tokenizer.word_counts
    print(wc)
    print(len(wc))

    model = create_model()
    model.summary()
    history = model.fit(train_seqs, train_labels, epochs=100, validation_data=(val_seqs, val_labels))

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')

    sentence = ["I'm really upset right now and not happy with you! ANGRY!",
                "She said yes! We're getting married! Wow!"]
    sequences = tokenizer.texts_to_sequences(sentence)
    print(sequences)
    padded = pad_sequences(sequences)
    print(model.predict(padded))

    # Since tokenizer is deprecated, let's try the TextVectorization layer
    vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_sequence_length=max_length)
    train_sentences, val_sentences, *_ = get_train_and_validation_sentences_and_labels(emotions_file_name)
    vectorization_layer.adapt(train_sentences)

    model = create_new_model(vectorization_layer)

    history = model.fit(np.array(train_sentences), train_labels, epochs=100, validation_data=(np.array(val_sentences), val_labels))

    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    print(model.predict(padded))





