from helpers import print_score
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

# test_data = pd.read_pickle('./data/test_data.pickle')
# X_test = test_data.loc[:, test_data.columns != 'label']
# Y_test = pd.read_pickle('./data/test_labels.pickle')

num_classes = len(np.unique(Y_train))

model = Sequential()
model.add(ZeroPadding2D((3,3)))

model.add(Conv2D(32, (3,3), strides=(1,1)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, to_categorical(Y_train, num_classes=10), epochs=5, batch_size=16)
score = model.evaluate(X_train, to_categorical(Y_train, num_classes=10))
print_score('Training Set:', score[0], score[1])
score = model.evaluate(X_valid, to_categorical(Y_valid, num_classes=10))
print_score('Validation Set:', score[0], score[1])
# score = model.evaluate(X_test, to_categorical(Y_test, num_classes=10))
# print('Test: ', score)

# plot_model(model, to_file='model3.png')


# Submission
# submit_data = pd.read_csv('./data/test.csv')
# submission = model.predict(submit_data.values)
# submission = np.argmax(submission, axis=1)
#
# df = pd.DataFrame({
#     'Label': submission,
# })
# df.index += 1
# df.to_csv('./data/submission.csv', index_label='ImageId')
