import pandas as pd
import numpy as np
from keras.utils import to_categorical

"""
    One vs Many Classifier
"""

train_data = pd.read_pickle('./data/train.pickle')

# Randomize Data
train_data = train_data.sample(frac=1).reset_index(drop=True)

def setup_data(data):
    X = data[['PetalLengthCm', 'PetalWidthCm']].values.T
    Y = data[['Species']].values.T
    Y = to_categorical(Y, num_classes=3)
    return X, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(X, Y):
    num_features = X.shape[0]
    num_labels = Y.shape[0]
    weights = np.random.randn(num_labels, num_features)
    biases = np.zeros(num_labels)
    return weights, biases

def forward(weights, X):
    """
    Cost function:
        -ylog(yhat) + (1-y)log(1-yhat)
    """
    # yhat =
    pass


def calc_cost(y, yhat):
    logsum = np.dot(-y, np.log(yhat)) + np.dot(1-y, np.log(1-yhat))
    return -1/m * logsum

def backward():
    pass

def update_weights():
    pass

def train(X, Y, learning_rate=0.01, num_iterations=100):
    weights = initialize_weights(X, Y)
    for i in range(num_iterations):
        forward()
        calc_cost()
        backward()
        weights = update_weights()
    return weights

def predict():
    pass

def accuracy():
    pass




X_train, Y_train = setup_data(train_data)

print(X_train[:][:1])
print(Y_train[:][:1])
