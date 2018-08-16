import pandas as pd

df = pd.read_csv('./data/test.csv')

# Transform to Categorical
df.Sex = pd.Categorical(df.Sex).codes

# Normalize - Age
mean = df[['Age', 'Pclass']].mean()
std = df[['Age', 'Pclass']].std()
df[['Age', 'Pclass']] = (df[['Age', 'Pclass']] - mean) / std

df = df[['Age', 'Pclass', 'Sex', 'PassengerId', 'Survived']]
df = df[pd.notnull(df['Age'])]

# Shuffle Data!
df = df.sample(frac=1).reset_index(drop=True)


# Split Data: Train, Validation, and Test Sets
train_size = 0.7
valid_size = 0.2
test_size = 0.0

m = df.shape[0]
prev = 0
train_data = df[prev:int(m * train_size)]
prev += int(m * train_size)
valid_data = df[prev:prev+int(m * valid_size)]
prev += int(m * valid_size)
test_data = df[prev:]

train_data.to_pickle('./data/train.pickle')
valid_data.to_pickle('./data/valid.pickle')
test_data.to_pickle('./data/test.pickle')
