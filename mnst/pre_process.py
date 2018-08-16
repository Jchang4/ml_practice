import pandas as pd

df = pd.read_csv('./data/train.csv')

print(df.head())

df = df.sample(frac=1).reset_index(drop=True)

df.loc[:, df.columns != 'label'] = df.loc[:, df.columns != 'label'].divide(255)

print(df.head())

train_size = 0.7
valid_size = 0.2
test_size = 0.1

m = df.shape[0]
prev = 0

train_data = df[:int(m * train_size)]
prev += int(m * train_size)
valid_data = df[prev:prev+int(m * valid_size)]
prev += int(m * valid_size)
test_data = df[prev:]

train_data.to_pickle('./data/train.pickle')
valid_data.to_pickle('./data/valid.pickle')
test_data.to_pickle('./data/test.pickle')
