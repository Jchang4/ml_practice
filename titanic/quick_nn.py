import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, test_data, losses):
        self.test_data = test_data
        self.losses = losses

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.losses.append(loss)



train_data = pd.read_pickle('./data/train.pickle')
X_train = train_data[['Age', 'Pclass', 'Sex']].values
Y_train = train_data[['Survived']].values

valid_data = pd.read_pickle('./data/valid.pickle')
X_valid = valid_data[['Age', 'Pclass', 'Sex']].values
Y_valid = valid_data[['Survived']].values


model = Sequential()
model.add(Dense(6, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

valid_loss = []

model.fit(X_train, Y_train, epochs=10, batch_size=50,
            callbacks=[TestCallback((X_valid, Y_valid), valid_loss)])

train_loss = model.history.history['loss']


score = model.evaluate(X_train, Y_train, batch_size=128)
print('Accuracy on Training Set:   {}'.format(round(score[1] * 100), 2))
score = model.evaluate(X_valid, Y_valid, batch_size=128)
print('Accuracy on Validation Set: {}'.format(round(score[1] * 100), 2))


print(train_loss)
print(valid_loss)


plt.plot(range(10), train_loss, 'r--',
        range(10), valid_loss, 'b')
plt.show()
