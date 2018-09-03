import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model


X_train = np.load('./data/train_data.pickle.npy')
Y_train = np.load('./data/train_labels.pickle.npy')

X_valid = np.load('./data/valid_data.pickle.npy')
Y_valid = np.load('./data/valid_labels.pickle.npy')


model = Sequential()

model.add(Flatten())

# model.add(Dense(1024, activation='sigmoid'))
# model.add(Dense(1024, activation='sigmoid'))
# model.add(Dense(512, activation='relu'))

# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(10, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, to_categorical(Y_train, num_classes=10), epochs=5, batch_size=16)

score = model.evaluate(X_train, to_categorical(Y_train, num_classes=10))
print('Train:', score)
score = model.evaluate(X_valid, to_categorical(Y_valid, num_classes=10))
print('Valid:', score)
