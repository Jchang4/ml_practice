import numpy as np
from activation_functions import sigmoid, sigmoid_backward, relu, relu_backward

"""
    General Model Structure:
        Linear->ReLU
        Linear->ReLU
            ...
        Linear->Sigmoid

    Last layer is a sigmoid layer for binary classification
"""

def initialize_parameters_deep(layer_dims):
    parameters = {}

    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(X, W, b):
    Z = np.dot(W, X) + b
    cache = (X, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propogate(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        caches.append(cache)

    # Last layer uses sigmoid
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches

def compute_cost(Y_pred, Y):
    m = Y.shape[1]
    cost = -1/m * (np.dot(Y, np.log(Y_pred.T)) + np.dot(1-Y, np.log(1-Y_pred.T)))

    cost = np.squeeze(cost)
    assert cost.shape == ()
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'relu':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propogate(AL, Y, caches):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1] # num. examples
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    # Lth layer, sigmoid -> linear
    current_cache = caches[L-1]
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')

    # L-1th layer to Layer1, relu -> linear
    for l in range(L-2, -1, -1):
        current_cache = caches[l]
        dA, dW, db = linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu')
        grads['dA' + str(l)] = dA
        grads['dW' + str(l+1)] = dW
        grads['db' + str(l+1)] = db

    return grads


def update_params(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]

    return parameters


def model(X, Y, hidden_layers, num_iterations=10000, learning_rate=0.01, print_cost=False):
    m = X.shape[1]
    n = X.shape[0]

    parameters = initialize_parameters_deep(hidden_layers)

    for i in range(num_iterations):
        AL, caches = forward_propogate(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propogate(AL, Y, caches)
        parameters = update_params(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print('Cost at iteration {}: {}'.format(i, cost))

    return parameters


def predict(X, parameters):
    predictions, caches = forward_propogate(X, parameters)
    return (predictions > 0.5)

def accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.shape[1] * 100


if __name__ == '__main__':
    import pandas as pd

    train_data = pd.read_pickle('./data/train.pickle')
    X_train = train_data[['Age', 'Pclass', 'Sex']].values.T
    Y_train = train_data[['Survived']].values.T

    valid_data = pd.read_pickle('./data/valid.pickle')
    X_valid = valid_data[['Age', 'Pclass', 'Sex']].values.T
    Y_valid = valid_data[['Survived']].values.T

    test_data = pd.read_pickle('./data/test_data.pickle')
    X_test = test_data[['Age', 'Pclass', 'Sex']].values.T
    # Y_test = test_data[['Survived']].values.T



    # --------------------------------------------------------------------------

    parameters = model(X_train, Y_train,
                        hidden_layers=[3,6,1],
                        num_iterations=15000,
                        learning_rate=0.005,
                        print_cost=True)
    pred = predict(X_train, parameters)
    print('Accuracy on Train: {}%'.format(accuracy(pred, Y_train)))
    pred = predict(X_valid, parameters)
    print('Accuracy on Valid: {}%'.format(accuracy(pred, Y_valid)))
    print()

    # --------------------------------------------------------------------------

    # parameters = model(X_train, Y_train,
    #                     hidden_layers=[3,9,1],
    #                     num_iterations=10000,
    #                     learning_rate=0.01,
    #                     print_cost=True)
    # pred = predict(X_train, parameters)
    # print('Accuracy on Train: {}%'.format(accuracy(pred, Y_train)))
    # pred = predict(X_valid, parameters)
    # print('Accuracy on Valid: {}%'.format(accuracy(pred, Y_valid)))


    # --------------------------------------------------------------------------

    # pred = predict(X_test, parameters)
    # submission = pd.DataFrame({
    #     'PassengerId': test_data['PassengerId'],
    #     'Survived': pred[0],
    # })
    # submission.Survived = pd.Categorical(submission.Survived).codes
    # submission.to_csv('./data/submission.csv', columns=['PassengerId', 'Survived'], index=False, index_label=False)
