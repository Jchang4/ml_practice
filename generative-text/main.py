import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense

def get_model(max_char_length=140):

    X_input = Input((max_char_length,))
    X = LSTM()(X_input)

    return Model(X_input, out)

if __name__ == '__main__':
    tokenized_text, char_to_idx = pickle.load(open('./data/processed-alice.pickle', 'rb'))
    X, Y = pickle.load(open('./data/alice-x-y-data.pickle', 'rb'))

    model = get_model()
    model.compile()
    model.fit(X, Y)