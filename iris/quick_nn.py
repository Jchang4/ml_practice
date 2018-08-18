import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

def accuracy(y_pred, y):
    return np.sum(y_pred == y) / y.shape[0] * 100

train_data = pd.read_pickle('./data/train.pickle')
X_train = train_data[['PetalLengthCm', 'PetalWidthCm']].values
Y_train = train_data[['Species']].values


valid_data = pd.read_pickle('./data/valid.pickle')
X_valid = valid_data[['PetalLengthCm', 'PetalWidthCm']].values
Y_valid = valid_data[['Species']].values

test_data = pd.read_pickle('./data/test.pickle')
X_test = test_data[['PetalLengthCm', 'PetalWidthCm']].values
Y_test = test_data[['Species']].values



model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(Y_train, num_classes=3)

# Train the model, iterating on the data in batches of 32 samples
model.fit(X_train, one_hot_labels, epochs=275, batch_size=32)
score = model.evaluate(X_valid, keras.utils.to_categorical(Y_valid, num_classes=3), batch_size=128)

print('Tranining Score:', score)

y_pred = np.argmax(model.predict(X_valid), axis=1)
y_pred = y_pred.reshape(Y_valid.shape[0], 1)
print('Valid Accuracy:', accuracy(y_pred, Y_valid), '%')

# y_pred = np.argmax(model.predict(X_test), axis=1)
# y_pred = y_pred.reshape(Y_test.shape[0], 1)
# print('Test Accuracy:', accuracy(y_pred, Y_test), '%')
