import re
import string

import numpy as np

from tensorflow.keras.utils import to_categorical


def get_vocab(sentences, verbose=False):
    all_words = set()
    for seq in sentences:
        for word in seq.split():
            if word not in all_words:
                all_words.add(word)

    all_words = sorted(list(all_words))
    num_words = len(all_words)

    word_index = dict([(word, i+1) for i, word in enumerate(all_words)])
    print(f"Vocabulary Size: {num_words}")
    return word_index


def get_cleaned_sentences(sentences, verbose=False):
    clean_sentences = []
    for input_seq in sentences:
        input_seq = input_seq.lower()
        input_seq = re.sub("'", "", input_seq)
        input_seq = ''.join(
            ch for ch in input_seq if ch not in set(string.punctuation))
        clean_sentences.append(input_seq)
    return clean_sentences


def get_longest_seq(sentences):
    length_list = []
    for seq in sentences:
        length_list.append(len(seq.split(' ')))
    max_length = np.max(length_list)
    print(f"Max Length Sentence: {max_length}")
    return max_length


def get_sequences_generator(sentences, batch_size):
    print(f"# Sentences: {len(sentences)}")
    sentences = get_cleaned_sentences(sentences)
    input_token_index = get_vocab(sentences)
    max_length = get_longest_seq(sentences)

    for j in range(0, len(sentences), batch_size):
        encoded_data = np.zeros((batch_size, max_length), dtype='float32')
        for i, input_text in enumerate(sentences[j:j+batch_size]):
            for t, word in enumerate(input_text.split()):
                encoded_data[i, t] = input_token_index[word]
    return encoded_data


def get_sequences(sentences, batch_size):
    print(f"# Sentences: {len(sentences)}")
    sentences = get_cleaned_sentences(sentences)
    input_token_index = get_vocab(sentences)
    max_length = get_longest_seq(sentences)
    encoded_data = np.zeros(
        (batch_size, max_length, len(input_token_index)+1), dtype='float32')

    for j in range(0, len(sentences), batch_size):
        for i, input_text in enumerate(sentences[j:j+batch_size]):
            for t, word in enumerate(input_text.split()):
                encoded_data[i, t] = to_categorical(
                    input_token_index[word], num_classes=len(input_token_index)+1)
    return sentences, encoded_data
