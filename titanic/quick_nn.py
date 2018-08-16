import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation


train_data = pd.read_pickle('./data/train.pickle')
X_train = train_data[['Age', 'Pclass', 'Sex']].values
Y_train = train_data[['Survived']].values

valid_data = pd.read_pickle('./data/valid.pickle')
X_valid = valid_data[['Age', 'Pclass', 'Sex']].values
Y_valid = valid_data[['Survived']].values


model = Sequential()
model.add(Dense(6, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=25, batch_size=32)
score = model.evaluate(X_valid, Y_valid, batch_size=128)

print(score)
