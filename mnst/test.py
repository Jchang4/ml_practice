import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_train_vaidate():
    data = pd.read_csv('./data/train.csv')
    data = data.sample(frac=1).reset_index(drop=True)

    labels = data['label']
    images = data[[i for i in data.columns if i != 'label']]

    images = images.values
    images = np.reshape(images, (images.shape[0], 28, 28, 1))

    train_data, valid_data, train_labels, valid_labels = train_test_split(images, labels, test_size=0.25)

    np.save('./data/train_data.pickle', train_data)
    np.save('./data/train_labels.pickle', train_labels)
    np.save('./data/valid_data.pickle', valid_data)
    np.save('./data/valid_labels.pickle', valid_labels)


if __name__ == '__main__':
    create_train_vaidate()
