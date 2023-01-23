import tensorflow as tf


if __name__ == '__main__':
    # Create some sample sentences from the wikipedia article on Henry the VIII.
    sentences = [   'Henry VIII was King of England from 21 April 1509 until his death.',
                    'He was the second Tudor monarch, succeeding his father, Henry VII.',
                    'Henry is best known for his six marriages, in particular his efforts to have his first marriage, to Catherine of Aragon, annulled.',
                    'His disagreement with the Pope on the question of such an annulment led Henry to initiate the English Reformation, separating the Church of England from papal authority.',
                    'He appointed himself the Supreme Head of the Church of England and dissolved convents and monasteries, for which he was excommunicated.']

    # Create the keras tokenizer
    num_words = 50
    oov_token = "<OOV>"
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token=oov_token)

    tokenizer.fit_on_texts(sentences)

    # Now display the vocabulary which is a dictionary of words and their index
    word_index = tokenizer.word_index
    print(f"Word index: {word_index}")

    # Let's create a sequence using an arbitrary sentence from the wikipedia article on George the II.
    test_sentence = "George II was King of Great Britain and Ireland from 11 June 1727 until his death."

    test_sequence = tokenizer.texts_to_sequences([test_sentence])
    print(f"Test sequence: {test_sequence}")

    # Now let's decode the sequence back to a sentence
    decoded_sentence = tokenizer.sequences_to_texts(test_sequence)
    print(f"Decoded sentence: {decoded_sentence}")

    # Or by using the index_word dictionary
    decoded_sentence = ' '.join([tokenizer.index_word[index] for index in test_sequence[0]])
    print(f"Decoded sentence: {decoded_sentence}")

    # Note that the decoded sentence will not be exactly the same as the original sentence because
    # the tokenizer will have removed punctuation and stopwords. Unknown words will be replaced with
    # the OOV token.
