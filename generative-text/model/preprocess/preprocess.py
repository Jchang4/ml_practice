import pickle
import numpy as np
from keras.utils import to_categorical

def get_char_mapping(s):
    """ Get characters mapped to idx, and vice versa """
    chars = sorted(list(set(s)))
    char_to_idx = dict((c,i) for i,c in enumerate(chars))
    idx_to_char = dict((i,c) for i,c in enumerate(chars))
    return char_to_idx, idx_to_char


def preprocess_text(text, save_path, max_sequence_length = 140):
    text = text.lower()
    text = text.replace('\n', '')

    char_to_idx, idx_to_char = get_char_mapping(text)

    # Prepare Data: change char to idx
    dataX = []
    dataY = []
    for i in range(0, len(text) - max_sequence_length):
        curr_text = text[i:i+max_sequence_length]
        seq_in = [char_to_idx[c] for c in curr_text]
        seq_out = char_to_idx[text[i+max_sequence_length]]
        dataX.append(seq_in)
        dataY.append(seq_out)

    # Prepare Data: reshape X and one-hot-encode Y
    num_samples = len(dataX)
    num_chars = len(char_to_idx)
    X = np.reshape(dataX, (num_samples, max_sequence_length, 1))
    X = X / float(num_chars)
    Y = to_categorical(dataY, num_classes=num_chars)

    # Prepare Data to be saved
    processed = {
        'X': X,
        'Y': Y,
        'dataX': dataX,
        'dataY': dataY,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
    }
    pickle.dump(processed, open(save_path, 'wb'))
















# alice = open('./data/alice.txt', 'r').read()
# alice = alice.lower()

# # Create Map from Char to Idx and vice versa
# unique_chars = sorted(list(set(alice)))
# char_to_idx = { c:i for i,c in enumerate(unique_chars) }
# idx_to_char = { i:c for i,c in enumerate(unique_chars) }

# # Prepare Data X and Y
# max_sequence_length = 100
# dataX = []
# dataY = []
# for i in range(0, len(alice) - max_sequence_length):
#     seq_in = alice[i:i+max_sequence_length]
#     seq_out = alice[i+max_sequence_length]
#     dataX.append([char_to_idx[c] for c in seq_in])
#     dataY.append(char_to_idx[seq_out])

# # Convert data to the shape: [samples, timesteps, features] and normalize
# # And change dataY to be one-hot encoded
# num_samples = len(dataX)
# X = np.reshape(dataX, (num_samples, max_sequence_length, 1))
# X = X / float(len(unique_chars))
# Y = to_categorical(dataY, num_classes=len(unique_chars))


# # Save X, Y, char_to_idx, idx_to_char
# processed_input = {
#     'X': X,
#     'Y': Y,
#     'dataX': dataX,
#     'dataY': dataY,
#     'char_to_idx': char_to_idx,
#     'idx_to_char': idx_to_char,
# }
# pickle.dump(processed_input, open('./data/processed-alice.pickle', 'wb'))