import numpy as np

class Neural:

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim, 1))
        b = 0
        return w, b

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def propogate(self, w, b, X, Y):
        """
        Calculate cost and gradient
        """
        m = X.shape[1]
        z = np.dot(w.T, X) + b
        A = self.sigmoid(z)
        cost = -1/m * np.sum(np.multiply(Y, np.log(A)), np.multiply(1-Y, np.log(1-A)))

        # Gradients
        Y_diff = A - Y
        dw = 1/m * np.dot(X, Y_diff.T)
        db = 1/m * np.sum(Y_diff)

        cost = np.squeeze(cost)
        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):

        costs = []

        for i in range(num_iterations):
            grads, cost = self.propogate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w -= np.multiply(learning_rate, dw)
            b -= np.multiply(learning_rate, db)

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

            params = {"w": w,
                      "b": b}

            grads = {"dw": dw,
                     "db": db}

            return params, grads, costs

    def predict(self, w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            Y_prediction[0][i] = 1 if A[0][i] >= 0.5 else 0

        return Y_prediction


    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.001, print_cost = False):

        w, b = self.initialize_with_zeros(X_train.shape[0])
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        w = parameters["w"]
        b = parameters["b"]

        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train" : Y_prediction_train,
             "w" : w,
             "b" : b,
             "learning_rate" : learning_rate,
             "num_iterations": num_iterations}

        return d


if __name__ == '__main__':
    import pandas as pd

    train_data = pd.read_pickle('./data/train.pickle')
    valid_data = pd.read_pickle('./data/valid.pickle')

    # Split Data into Features & Labels
    X_train = train_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values.T
    Y_train = train_data[['Species']].values.T

    X_valid = valid_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values.T
    Y_valid = valid_data[['Species']].values.T

    print(X_train[:][1:5])
    print(Y_train[:][1:5])
    print(X_train.shape)
    print(Y_train.shape)
    print()

    clf = Neural()
    d = clf.model(X_train, Y_train, X_valid, Y_valid, print_cost = True)
