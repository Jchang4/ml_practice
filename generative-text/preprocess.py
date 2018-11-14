"""
    - Make everything lowercase to reduce vocabulary

    - Create Word Mapping (word to index)
"""
import pickle

def preprocess(max_sentence_length=27):
    text = open('./data/alice.txt', 'r').read()
    text = text.lower()

    # Create Word Mapping
    words = sorted(list(set(text.split())))
    words_to_idx = dict((w,i) for i,w in enumerate(words))

    # Break text into sentences
    tokenizer = pickle.load(open('./data/sentence-tokenizer.pickle', 'rb'))
    text = tokenizer.tokenize(text)

    # Shorten / Pad sentences to be same length
    tokenized_text = []
    for sentence in text:
        words = sentence.split()
        if len(words) > max_sentence_length:
            words = words[:max_sentence_length]
        elif len(words) < max_sentence_length:
            words = words + ([-1] * (max_sentence_length - len(words)))
        
        # Tokenize words - i.e. return list of idxs instead of words        
        tokenized_words = [words_to_idx[w] if w != -1 else -1 for w in words]
        tokenized_text.append(tokenized_words)
    
    return tokenized_text, words_to_idx


if __name__ == '__main__':
    tokenized_text, words_to_idx = preprocess()

    pickle.dump([tokenized_text, words_to_idx], open('./data/processed-alice.pickle', 'wb'))