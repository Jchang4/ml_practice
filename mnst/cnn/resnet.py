from helpers import print_score
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Add
from keras.layers import Flatten, ZeroPadding2D, BatchNormalization, Activation
from keras.layers import Conv2D, Dense, AveragePooling2D, MaxPooling2D
from keras.initializers import glorot_uniform


X_train = np.load('./data/train_data.pickle.npy')
Y_train = np.load('./data/train_labels.pickle.npy')


# valid_data = pd.read_pickle('./data/valid_data.pickle')
X_valid = np.load('./data/valid_data.pickle.npy')
Y_valid = np.load('./data/valid_labels.pickle.npy')


def identity_network(X, f, filters, stage, block):
    conv_name = 'res_{}_{}_branch'.format(str(stage), block)
    bn_name = 'bn_{}_{}_branch'.format(str(stage), block)

    # Filters
    F1, F2, F3 = filters

    # Save for Skip Connection
    X_shortcut = X

    # 3 layers -> Conv2D, BatchNormalization, Activation('relu')
    X = Conv2D(F1, (1,1), strides = (1,1), padding = 'valid', name = conv_name+'2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f,f), strides = (1,1), padding = 'same', name = conv_name+'2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name+'2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name+'2c')(X)

    # Add Skip Connection
    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    return X

def convolutional_network(X, f, filters, stage, block, s = 2):
    conv_name = 'res_{}_{}_branch'.format(str(stage), block)
    bn_name = 'bn_{}_{}_branch'.format(str(stage), block)

    # Filters
    F1, F2, F3 = filters

    # Save for Skip Connection
    X_shortcut = X

    # 3 layers -> Conv2D, BatchNormalization, Activation('relu')
    X = Conv2D(F1, (1,1), strides = (s,s), padding = 'valid', name = conv_name+'2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name+'2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f,f), strides = (1,1), padding = 'same', name = conv_name+'2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1,1), strides = (1,1), padding = 'valid', name = conv_name+'2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name+'2c')(X)

    # Add Skip Connection
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), padding='valid', name = conv_name + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name = bn_name + '1')(X_shortcut)

    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    return X


def ResNet50(input_shape = (28, 28, 1), num_classes = 10):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)

    # Stage 1
    X = Conv2D(64, (7,7), strides = (1,1), padding = 'same', name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    # Stage 2
    X = convolutional_network(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', s = 1)
    X = identity_network(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'b')
    X = identity_network(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'c')

    # Stage 3
    X = convolutional_network(X, f = 3, filters = [128,128,512], stage = 3, block = 'a', s = 2)
    X = identity_network(X, f = 3, filters = [128,128,512], stage = 3, block = 'b')
    X = identity_network(X, f = 3, filters = [128,128,512], stage = 3, block = 'c')
    X = identity_network(X, f = 3, filters = [128,128,512], stage = 3, block = 'd')

    # Stage 4
    X = convolutional_network(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s = 2)
    X = identity_network(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'b')
    X = identity_network(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'c')
    X = identity_network(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'd')
    X = identity_network(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'e')
    X = identity_network(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'f')


    # Stage 5
    X = convolutional_network(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', s = 2)
    X = identity_network(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'b')
    X = identity_network(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'c')

    # X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)
    X = MaxPooling2D((2,2))(X)

    # FC
    X = Flatten()(X)
    X = Dense(num_classes, activation='softmax', name='fc' + str(num_classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model


model = ResNet50()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, to_categorical(Y_train, num_classes=10), epochs=2, batch_size=32)
score = model.evaluate(X_train, to_categorical(Y_train, num_classes=10))
print_score('Training Set:', score[0], score[1])
score = model.evaluate(X_valid, to_categorical(Y_valid, num_classes=10))
print_score('Validation Set:', score[0], score[1])
