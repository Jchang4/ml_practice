"""
    - Make everything lowercase to reduce vocabulary

    - Create Word Mapping (word to index)
"""
import pickle
import numpy as np

def preprocess(max_char_length=140):
    text = open('./data/alice.txt', 'r').read()
    text = text.lower()

    # Create Word Mapping
    chars = set(text)
    chars.add('PAD')
    chars = sorted(list(chars))
    char_to_idx = dict((c,i) for i,c in enumerate(chars))

    # Break text into sentences
    tokenizer = pickle.load(open('./data/sentence-tokenizer.pickle', 'rb'))
    text = tokenizer.tokenize(text)

    # Shorten / Pad sentences to be same length
    tokenized_text = []
    for sentence in text:
        curr_chars = list(sentence)
        if len(curr_chars) > max_char_length:
            curr_chars = curr_chars[:max_char_length]
        elif len(curr_chars) < max_char_length:
            curr_chars = curr_chars + (['PAD'] * (max_char_length - len(curr_chars)))
        
        # Tokenize characters - i.e. return list of idxs instead of characters
        tokenized_chars = []
        for c in curr_chars:
            one_hot = [0] * len(chars)
            idx = char_to_idx[c]
            one_hot[idx] = 1
            tokenized_chars.append(np.array(one_hot))
        tokenized_text.append(np.array(tokenized_chars))
    
    tokenized_text = np.array(tokenized_text)
    
    return tokenized_text, char_to_idx

def get_X_and_Y(X):
    """ Y is X shifted over 1 time-step """
    Y = np.roll(X, -1)
    return X, Y

if __name__ == '__main__':
    tokenized_text, char_to_idx = preprocess()
    X, Y = get_X_and_Y(tokenized_text)

    pickle.dump([tokenized_text, char_to_idx], open('./data/processed-alice.pickle', 'wb'))
    pickle.dump([X, Y], open('./data/alice-x-y-data.pickle', 'wb'))