import pickle
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense

vocab = list('abcdefghijklmnopqrstuvwxyz!?.')
char_to_idx = dict((c,i) for i,c in enumerate(vocab))
idx_to_char = dict((str(i),c) for i,c in enumerate(vocab))

tokenized_text, char_to_idx, idx_to_char = pickle.load(open('./data/processed-alice.pickle', 'rb'))

print(tokenized_text[0][5])

# def get_model():
#     x0 = Input(shape=())
#     a0
#     c0

#     return Model(inputs, outputs)