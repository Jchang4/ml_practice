import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint

# Get Pre-Processed Data
X, Y, char_to_idx, idx_to_char = pickle.load(open('./data/processed-alice.pickle', 'rb'))


def get_model_checkpoint(filename = './data/weights-{epoch:02d}-{loss:02d}.h5'):
    return ModelCheckpoint(filename,
                            save_best_only=1,
                            verbose=1,
                            mode='min')

def get_model(n_samples, n_timesteps, n_features, n_a):
    checkpoint = get_model_checkpoint()

    x0 = Input(shape=(n_timesteps, 1))
    a0 = Input(shape=(n_samples, n_a))
    c0 = Input(shape=(n_samples, n_a))
    a = a0
    c = c0
    outputs = []



    return Model([x0, a0, c0], outputs, callbacks=[checkpoint])