import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    """
    Given size of input, hidden, and output layer

    Returns:
        parameters: {
            W1
            b1
            W2
            b2
        }
    """

    W1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, Z1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}

    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]

    logprods = np.add(np.multiply(np.log(A2), Y), np.multiply(np.log(1-A2), 1-Y))
    cost = -1/m * np.sum(logprods)

    cost = np.squeeze(cost)

    assert(isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A2 = cache['A2']
    A1 = cache['A1']

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dZ2': dZ2,
             'dW2': dW2,
             'db2': db2,
             'dZ1': dZ1,
             'dW1': dW1,
             'db1': db1}

    return grads

def update_parameters(parameters, grads, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = np.subtract(W1, learning_rate * dW1)
    b1 = np.subtract(b1, learning_rate * db1)
    W2 = np.subtract(W2, learning_rate * dW2)
    b2 = np.subtract(b2, learning_rate * db2)

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, learning_rate=0.01, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(num_iterations):
        A2, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A2, Y_train)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print('Cost at {}: {}'.format(i, cost))

    return parameters

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    return (A2 > 0.5)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from planar_utils import plot_decision_boundary

    train_data = pd.read_pickle('./data/train.pickle')
    X_train = train_data[['SepalLengthCm', 'SepalWidthCm']].values.T
    Y_train = train_data[['Species']].values.T

    valid_data = pd.read_pickle('./data/valid.pickle')
    X_valid = valid_data[['SepalLengthCm', 'SepalWidthCm']].values.T
    Y_valid = valid_data[['Species']].values.T

    print()
    print('X_train shape:', X_train.shape)
    print(X_train[:,:5])
    print('Y_train shape:', Y_train.shape)
    print(Y_train[:,:5])
    print()

    parameters = nn_model(X_train, Y_train, 4, num_iterations=10000, print_cost=True)

    predictions = predict(parameters, X_train)
    print('Accuracy: {}%'.format( np.sum(predictions == Y_train) / float(Y_train.size) * 100))
    predictions = predict(parameters, X_valid)
    print('Accuracy: {}%'.format( np.sum(predictions == Y_valid) / float(Y_valid.size) * 100))

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X_train, Y_train.reshape(Y_train.shape[1]))
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()
