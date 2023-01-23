import tensorflow as tf
import pickle

from ml_utils import print_vocabulary_of_tokenizer, visualize_embeddings_from_model, plot_word_count_distribution

# Load the model
model = tf.keras.models.load_model('c3-w3-lab6-sarcasm-with-1d-conv.h5')

# Load the tokenizer using pickle
tokenizer = None
with open('c3-w3-lab6-sarcasm-with-1d-conv-tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Get the embedding layer
embedding_layer = None
for layer in model.layers:
    if type(layer) == tf.keras.layers.Embedding:
        embedding_layer = layer

# Print the vocabulary of the tokenizer
print_vocabulary_of_tokenizer(tokenizer)

# Print the size of the vocabulary
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# Extract the files from the embedding layer and print their names
print(visualize_embeddings_from_model(model, embedding_layer.name, tokenizer))

# Print word count distribution
print(plot_word_count_distribution(tokenizer, 0, tokenizer.num_words, False))
# Print the top 10 words
print(plot_word_count_distribution(tokenizer, 0, 10, True))

print("Done")