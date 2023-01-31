import os
import pickle
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import ml_utils

vocab_size = 10000
max_length = 120
embedding_dim = 16
trunc_type='post'
oov_tok = "<OOV>"

def create_and_train_model():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    # Print the features of the dataset
    for key, value in info.features.items():
        print(key, value)

    # Get the number of classes from the dataset info
    # The number of neurons in the output layer will be equal to the number of classes
    # But if the number of classes is 2, then the model will have 1 neuron in the output layer with a sigmoid activation function
    # if the number of classes is greater than 2, then the model will have the number of classes neurons in the output layer with a softmax activation function
    num_classes = info.features['label'].num_classes
    print(f"Number of classes: {num_classes}")

    # Print the splits of the dataset
    for key, value in info.splits.items():
        print(key, value)

    # Display the content of the first data of the dataset and the corresponding label
    # in order to find out whether the label 0 is negative or positive
    for example in imdb['train'].take(1):
        # Print the text
        print(example[0].numpy())
        # Print the label
        print(f"Label: {example[1].numpy()}")

    train_data, test_data = imdb['train'], imdb['test']

    # Initialize sentences and labels lists
    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    for s,l in train_data:
      training_sentences.append(s.numpy().decode('utf8'))
      training_labels.append(l.numpy())

    for s,l in test_data:
      testing_sentences.append(s.numpy().decode('utf8'))
      testing_labels.append(l.numpy())

    # Convert the lists to numpy arrays
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    # Initialize the Tokenizer class
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary for the training sentences
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Generate and pad the training sequences
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    # Generate and pad the test sequences
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length, truncating=trunc_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        # If the number of classes is 2 which means binary classification, then the activation function of the output layer will be sigmoid
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Setup the training parameters
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model.summary()

    num_epochs = 10
    model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

    # Save the model
    model.save("simple-model.h5")

    # Save the tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, tokenizer

# If the model is already trained, then load the model
if os.path.exists("simple-model.h5"):
    model = tf.keras.models.load_model("simple-model.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
else:
    model, tokenizer = create_and_train_model()

# Now, let's predict the sentiment of a new review from the folder sample_reviews:
for filename in os.listdir("sample_reviews"):
    with open(os.path.join("sample_reviews", filename), "r") as f:
        review = f.read()
        review = ml_utils.pre_process_text(review)

        # Print the review
        print(review)

        # Generate the sequence for the review
        sequence = tokenizer.texts_to_sequences([review])
        # Pad the sequence
        padded = pad_sequences(sequence, maxlen=max_length, truncating=trunc_type)

        # The max length of the sequence is 120. But the content of the file is much longer than 120 words.
        # So the sequence will be truncated to 120 words. Which leads to a loss of information and
        # the model will not be able to predict the sentiment of the review correctly.
        # Predict the sentiment
        sentiment = model.predict(padded)
        print(f"{filename}: {sentiment}")

        # Let's take the last 120 words of the review and predict the sentiment
        review = review.split(" ")
        review = " ".join(review[-120:])
        print(review)
        # Generate the sequence for the review
        sequence = tokenizer.texts_to_sequences([review])
        # Pad the sequence
        padded = pad_sequences(sequence, maxlen=max_length, truncating=trunc_type)
        sentiment = model.predict(padded)
        print(f"{filename}: {sentiment}")



