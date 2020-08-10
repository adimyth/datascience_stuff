import functools
import math
import pickle
import re
import string
import numpy as np
import requests
from tensorflow.keras.models import load_model

from configs import config

# CONSTANTS
max_length_src = config["max_length_src"]
max_length_tar = config["max_length_tar"]

print("[INFO] Loading Word Indexes ...")
with open(config["input_word_index"], "rb") as file:
    input_token_index = pickle.load(file)
with open(config["target_word_index"], "rb") as file:
    target_token_index = pickle.load(file)
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())
print("[INFO] Loading Encoder & Decoder ...")
encoder_model = load_model(config["encoder_path"])
decoder_model = load_model(config["decoder_path"])


def clean(input_seq):
    input_seq = input_seq.lower()
    input_seq = re.sub("'", "", input_seq)
    input_seq = "".join(ch for ch in input_seq if ch not in set(string.punctuation))
    input_seq = input_seq.strip()
    return input_seq


def get_input_seq(input_seq):
    input_seq = clean(input_seq)
    encoder_input_data = np.zeros((1, max_length_src), dtype="float32")
    for t, word in enumerate(input_seq.split()):
        encoder_input_data[0, t] = input_token_index[word]
    return encoder_input_data


@functools.lru_cache(maxsize=128)
def decode_sequence(input_seq):
    input_seq = get_input_seq(input_seq)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    # use `START_` as the first character
    target_seq[0, 0] = target_token_index["START_"]

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sampling a token with max probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += " " + sampled_char

        # Exit condition: either hit max length or find stop character.
        if sampled_char == "_END" or len(decoded_sentence) > max_length_tar:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def beam_search_decoder(predictions, top_k):
    # start with an empty sequence with zero score
    output_sequences = [([], 0)]

    # looping through all the predictions
    for token_probs in predictions:
        new_sequences = []

        # append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                # considering log-likelihood for scoring
                new_score = old_score + math.log(token_probs[char_index])
                new_sequences.append((new_seq, new_score))

        # sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)

        # select top-k based on score
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]

    return output_sequences


@functools.lru_cache(maxsize=128)
def decode_sequence_beam_search(input_seq, beam_width=3):
    probabilities = []
    # Encode the input as state vectors.
    input_seq = get_input_seq(input_seq)
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index["START_"]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sampling a token with max probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        probabilities.append(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += " " + sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "_END" or len(decoded_sentence) > max_length_tar:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    # storing multiple results
    outputs = []
    beam_search_preds = beam_search_decoder(probabilities, top_k=beam_width)
    for prob_indexes, score in beam_search_preds:
        decoded_sentence = ""
        for index in prob_indexes:
            sampled_char = reverse_target_char_index[index]
            decoded_sentence += " " + sampled_char
            if sampled_char == "_END" or len(decoded_sentence) > max_length_tar:
                break
        outputs.append(decoded_sentence)
    return outputs
