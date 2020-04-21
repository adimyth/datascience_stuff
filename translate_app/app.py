from configs import config
import numpy as np
import pickle
import re
import requests
import string

# CONSTANTS
max_length_src = config["max_length_src"]
max_length_tar = config["max_length_tar"]

print("Loading word indexes ....")
with open(config["input_word_index"], "rb") as file:
    input_token_index = pickle.load(file)
with open(config["target_word_index"], "rb") as file:
    target_token_index = pickle.load(file)
reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())


def clean(input_seq):
    input_seq = input_seq.lower()
    input_seq = re.sub("'", "", input_seq)
    input_seq = ''.join(ch for ch in input_seq if ch not in set(string.punctuation))
    input_seq = input_seq.strip()
    return input_seq

def get_input_seq(input_seq):
    input_seq = clean(input_seq)
    encoder_input_data = np.zeros((1, max_length_src), dtype='float32')
    for t, word in enumerate(input_seq.split()):
        encoder_input_data[0, t] = input_token_index[word]
    return encoder_input_data


def decode_sequence(model, encoder_model, decoder_model, input_seq):
    input_seq = get_input_seq(input_seq)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))
    # use `START_` as the first character
    target_seq[0, 0] = target_token_index['START_']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sampling a token with max probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > max_length_tar):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

