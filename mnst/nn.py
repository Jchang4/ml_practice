import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation


train_data = pd.read_pickle('./data/train.pickle')
X_train = train_data.loc[:, train_data.columns != 'label']
Y_train = train_data['label']

valid_data = pd.read_pickle('./data/valid.pickle')
X_valid = valid_data.loc[:, valid_data.columns != 'label']
Y_valid = valid_data['label']

test_data = pd.read_pickle('./data/test.pickle')
X_test = test_data.loc[:, test_data.columns != 'label']
Y_test = test_data['label']

model = Sequential()
model.add(Dense(785*2, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(785*2, activation='relu'))
model.add(Dense(785*2, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


one_hot_labels = to_categorical(Y_train, num_classes=10)

model.fit(X_train, one_hot_labels, epochs=10, batch_size=32)
score = model.evaluate(X_valid, to_categorical(Y_valid, num_classes=10))
print('Valid:', score)
score = model.evaluate(X_test, to_categorical(Y_test, num_classes=10))
print('Test: ', score)



# Submission
submit_data = pd.read_csv('./data/test.csv')
submission = model.predict(submit_data.values)
submission = np.argmax(submission, axis=1)

df = pd.DataFrame({
    'Label': submission,
})
df.index += 1
df.to_csv('./data/submission.csv', index_label='ImageId')
