"""
    - Make everything lowercase to reduce vocabulary

    - Create Word Mapping (word to index)
"""
import pickle
import numpy as np
from keras.utils.np_utils import to_categorical

def preprocess(max_char_length=140):
    text = open('./data/alice.txt', 'r').read()
    text = text.lower()
    text = text.replace('\n', ' ')

    # Create Word Mapping
    chars = set(text)
    chars.add('PAD')
    chars = sorted(list(chars))
    char_to_idx = dict((c,i) for i,c in enumerate(chars))
    idx_to_char = dict((str(i),c) for i,c in enumerate(chars))

    # Break text into sentences
    text = [text[i:i+max_char_length] for i in range(0, len(text), max_char_length)]

    # Shorten / Pad sentences to be same length
    tokenized_text = []
    for sentence in text:
        curr_chars = list(sentence)
        if len(curr_chars) > max_char_length:
            curr_chars = curr_chars[:max_char_length]
        elif len(curr_chars) < max_char_length:
            curr_chars = curr_chars + (['PAD'] * (max_char_length - len(curr_chars)))
        
        # Tokenize characters - i.e. return list of idxs instead of characters
        tokenized_chars = to_categorical([char_to_idx[c] for c in curr_chars], num_classes=len(chars))
        tokenized_text.append(np.array(tokenized_chars))
    
    tokenized_text = np.array(tokenized_text)
    
    return tokenized_text, char_to_idx, idx_to_char

def get_X_and_Y(X):
    """ Y is X shifted over 1 time-step """
    Y = np.roll(X, -1, axis=1)
    return X, Y

if __name__ == '__main__':
    tokenized_text, char_to_idx, idx_to_char = preprocess()
    X, Y = get_X_and_Y(tokenized_text)
    
    pickle.dump([tokenized_text, char_to_idx, idx_to_char], open('./data/processed-alice.pickle', 'wb'))
    pickle.dump([X, Y], open('./data/alice-x-y-data.pickle', 'wb'))