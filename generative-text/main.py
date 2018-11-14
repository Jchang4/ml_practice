import pickle

tokenized_text, words_to_idx = pickle.load(open('./data/processed-alice.pickle', 'rb'))


print(words_to_idx)