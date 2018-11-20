import pickle
from preprocess import preprocess_text

text = open('./data/alice/alice-in-wonderland.txt', 'r').read()
preprocess_text(text, './data/alice/alice-processed.pickle')