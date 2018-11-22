from preprocess import preprocess_text

text = open('./data/shakespeare/sonnets.txt', 'r').read()

preprocess_text(text, './data/shakespeare/processed.pickle')