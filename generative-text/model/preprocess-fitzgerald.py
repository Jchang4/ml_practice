from preprocess import preprocess_text

book_paths = [
    './data/fitzgerald/beautiful-and-damned.txt',
    './data/fitzgerald/flappers-and-philosophers.txt',
    './data/fitzgerald/tales-of-the-jazz-age.txt',
    './data/fitzgerald/this-side-of-paradise.txt',
]

# Combine books into 1 big book
combined_text = ''
combined_len = 0
for path in book_paths:
    txt = open(path, 'r').read()
    combined_len = combined_len + len(txt)
    combined_text = combined_text + ' ' + txt

preprocess_text(combined_text, '../data/fitzgerald/processed-all-books.pickle')

