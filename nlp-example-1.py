import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from ml_utils import pre_process_text

def create_model(tokenizer):
    embedding_dim = 64
    lstm1_dim = 64
    lstm2_dim = 32
    dense_dim = 64

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm1_dim, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm2_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Print the model summary
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

plain_imdb, plain_info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
# Print all the features of the dataset
# The text feature of the dataset is a tf.string because the dataset is a plain text dataset.
for key, value in plain_info.features.items():
    print(key, value)

sub_word_imdb, sub_word_info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
# Print all the features of the dataset
# Note that the text feature of the dataset contains an encoder
# The text feature has the dtype int64 because the encoder is a subword encoder
for key, value in sub_word_info.features.items():
    print(key, value)

# Get the encoder which is used to encode the text
sub_word_tokenizer = sub_word_info.features['text'].encoder
print(sub_word_tokenizer.subwords[:10])

# print the first data of the plain text dataset
# It will be a tuple of (text, label)
for example in plain_imdb['train'].take(1):
    print(example)

# print the first data of the subword text dataset
# It will not contain text, but instead it will contain the encoded text as a vector of integers
for example in sub_word_imdb['train'].take(1):
    print(example)
    # Let's decode the encoded text
    decoded_text = sub_word_tokenizer.decode(example[0])
    print(decoded_text)

BUFFER_SIZE = 10000
BATCH_SIZE = 256

dataset = sub_word_imdb

# Print the number of examples in each split
print(sub_word_info.splits['train'].num_examples)
print(sub_word_info.splits['test'].num_examples)

# Get the train and test splits
train_data, test_data = dataset['train'], dataset['test']

# Shuffle the training data
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

model = create_model(sub_word_tokenizer)

# Only train the model if the modek has not yet been saved
if not os.path.exists('model-nlp-example-1.h5'):
    # Train the model
    model.fit(train_dataset, epochs=10, validation_data=test_dataset)

    # Save the model
    model.save('model-nlp-example-1.h5')
else:
    # Load the model
    model = tf.keras.models.load_model('model-nlp-example-1.h5')

# Now predict the sentiment of an arbitrary review from the validation dataset
for example in test_dataset.take(1):
    # Get a random review from the validation dataset
    review_index = np.random.randint(0, BATCH_SIZE)

    # No padding or truncation is needed because the review is already padded and truncated
    review, label = example[0][review_index], example[1][review_index]

    array = review.numpy()
    # Print the length of the review and count the number of zero elements from the end which is padding
    print(f"Length of review: {len(array)}")
    print(f"Number of padding elements: {len(array) - np.count_nonzero(array)}")

    # Print the zero elements from the end
    # Get the current print options threshold and set it to a very high value so that the entire array is printed
    threshold = np.get_printoptions()['threshold']
    np.set_printoptions(threshold=np.inf)
    print(f"Padding elements: {array[:np.count_nonzero(array) + 2]}")
    # Reset the print options threshold
    np.set_printoptions(threshold=threshold)

    # Decode the review
    decoded_review = sub_word_tokenizer.decode(review)
    # Predict the sentiment of the review
    prediction = model.predict(tf.expand_dims(review, 0))
    print(prediction)
    print('Review: {}'.format(decoded_review))
    print('Sentiment: {}'.format('Positive' if prediction[0][0] > 0.5 else 'Negative'))

# Now predict the sentiment of an arbitrary review from the folder sample_reviews
for filename in os.listdir('sample_reviews'):
    with open(os.path.join('sample_reviews', filename), 'r') as f:

        # Read the review from the file
        review = f.read()

        # Preprocess the review
        review = pre_process_text(review)
        print(f"Pre processed review: {review}")

        # Encode the review
        encoded_review = sub_word_tokenizer.encode(review)

        # Decode the review again to check if the encoding and decoding works
        decoded_review = sub_word_tokenizer.decode(encoded_review)
        print(f"Decoded review: {decoded_review}")

        # Predict the sentiment of the review
        prediction = model.predict(tf.expand_dims(encoded_review, 0))
        print('Sentiment: {}'.format('Positive' if prediction[0][0] > 0.5 else 'Negative'))