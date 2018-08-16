import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/train.csv')
df = df.sample(frac=1).reset_index(drop=True)

labels = df['label']
df.drop(labels='label', axis=1)
data = df

data /= 255.0


X_rest, X_test, y_rest, y_test = train_test_split(data, labels, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_rest, y_rest, test_size=0.2)


X_train.to_pickle('./data/train_data.pickle')
X_valid.to_pickle('./data/valid_data.pickle')
X_test.to_pickle('./data/test_data.pickle')

y_train.to_pickle('./data/train_labels.pickle')
y_valid.to_pickle('./data/valid_labels.pickle')
y_test.to_pickle('./data/test_labels.pickle')
