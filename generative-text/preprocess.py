"""
    - Make everything lowercase to reduce vocabulary

    - Create Word Mapping (word to index)
"""
import pickle

def preprocess(max_char_length=140):
    text = open('./data/alice.txt', 'r').read()
    text = text.lower()

    # Create Word Mapping
    chars = sorted(list(set(text)))
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
            curr_chars = curr_chars + ([-1] * (max_char_length - len(curr_chars)))
        
        # Tokenize characters - i.e. return list of idxs instead of characters        
        tokenized_chars = [char_to_idx[c] if c != -1 else -1 for c in curr_chars]
        tokenized_text.append(tokenized_chars)
    
    return tokenized_text, char_to_idx


if __name__ == '__main__':
    tokenized_text, char_to_idx = preprocess()

    pickle.dump([tokenized_text, char_to_idx], open('./data/processed-alice.pickle', 'wb'))