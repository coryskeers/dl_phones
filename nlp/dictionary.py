import pandas as pd
import numpy as np

import re

class Dictionary:
    def __init__(self):
        self.dict_filename = 'cmudict.txt'
        self.symbols_filename = 'cmudict_symbols.txt'
        self.phones_filename = 'cmudict_phones.txt'
        self.build_phonemes()
        self.build_phoneme_defs()
        self.build_symbols

    def build_phonemes(self):
        self.phoneme_dict = {}
        self.dict_phoneme = {}
        print("Building phoneme list...")
        with open(self.dict_filename, encoding = 'latin-1') as infile:
            phoneme_list = [line.strip().split('  ', 1) for line in infile if line[0] != ';']
        print("Phoneme list complete. Building dict...")
        for entry in phoneme_list:
            word = entry[0]
            self.phoneme_dict[word] = entry[1].split()
            self.dict_phoneme[entry[1]] = word
        print("Phoneme dict complete.")

    def build_phoneme_defs(self):
        print("Getting phoneme definitions...")
        self.phoneme_defs = {}
        with open(self.phones_filename) as infile:
            for line in infile:
                split = line.split()
                self.phoneme_defs[split[0]] = split[1]
        print("Phoneme definitions complete.")

    def build_symbols(self):
        print("Building phoneme symbols...")
        with open(self.symbols_filename) as infile:
            self.phoneme_list = [line.strip() for line in infile]
        print("Phoneme symbols complete.")

    def sent_to_phonemes(self, sentence):
        stripped = re.sub(r'''([?!.",;:]+)''', ' \\1 ', sentence.upper())
        words = stripped.split()
        phonemes = ""
        for word in words:
            if word in self.phoneme_dict:
                phonemes = phonemes + ' ' + ' '.join(self.phoneme_dict[word])
        return phonemes.strip()

class Corpus:
    def __init__(self):
        pass

#####

def add_word_noise(words, threshold = 1.5):
    s = "abcdefghijklmnopqrstuvwxyz"
    res = []
    for word in words:
        if np.random.randn() > threshold:
            i = np.random.randint(0, len(word))
            char = s[np.random.randint(0,26)]
            word = word[:i] + char + word[i:]
        if np.random.randn() > threshold:
            i = np.random.randint(0, len(word) - 1)
            listed = list(word)
            listed[i], listed[i + 1] = listed[i + 1], listed[i]
            word = ''.join(listed)
        if np.random.randn() > threshold:
            i = np.random.randint(0, len(word))
            if i == len(word) - 1:
                word = word[:-1]
            else:
                word = word[:i] + word[i + 1:]
        res.append(word)
    return res

def dl():
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense

    input_sentences = []
    target_pronunciations = []

    input_chars = set()
    target_chars = set()

    batch_size = 64
    epochs = 100
    latent_dim = 256
    num_samples = 10000

    max_encoder_sequence_length = 100
    max_decoder_sequence_length = 100

    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    
    encoder_input_data = np.zeros(
        (len(input_sentences), max_encoder_sequence_length, num_encoder_tokens),
        dtype = 'float32')
    decoder_input_data = np.zeros(
        (len(input_sentences), max_decoder_sequence_length, num_decoder_tokens),
        dtype = 'float32')
    decoder_target_data = np.zeros(
        (len(input_sentences), max_decoder_sequence_length, num_decoder_tokens),
        dtype = 'float32')

    ####
    ####

    encoder_inputs = Input(shape = (None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state = True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape = (None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequence = True, return_state = True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation = "softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size = batch_size,
              epochs = epochs,
              validation_split = 0.2)
    model.save('seq2seq.h5')


    ###
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape = (latent_dim,))
    decoder_state_input_c = Input(shape = (latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state = decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
