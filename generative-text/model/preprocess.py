import pickle
import numpy as np
from keras.utils import to_categorical

alice = open('./data/alice.txt', 'r').read()
alice = alice.lower()

# Create Map from Char to Idx and vice versa
unique_chars = sorted(list(set(alice)))
char_to_idx = { c:i for i,c in enumerate(unique_chars) }
idx_to_char = { i:c for i,c in enumerate(unique_chars) }

# Prepare Data X and Y
max_sequence_length = 100
dataX = []
dataY = []
for i in range(0, len(alice) - max_sequence_length):
    seq_in = alice[i:i+max_sequence_length]
    seq_out = alice[i+max_sequence_length]
    dataX.append([char_to_idx[c] for c in seq_in])
    dataY.append(char_to_idx[seq_out])

# Convert data to the shape: [samples, timesteps, features] and normalize
# And change dataY to be one-hot encoded
num_samples = len(dataX)
X = np.reshape(dataX, (num_samples, max_sequence_length, 1))
X = X / float(len(unique_chars))
Y = to_categorical(dataY, num_classes=len(unique_chars))


# Save X, Y, char_to_idx, idx_to_char
processed_input = {
    'X': X,
    'Y': Y,
    'dataX': dataX,
    'dataY': dataY,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}
pickle.dump(processed_input, open('./data/processed-alice.pickle', 'wb'))