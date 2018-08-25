import pandas as pd

df = pd.read_csv('./data/train.csv')
df = df.sample(frac=1).reset_index(drop=True)


# Drop unused columns
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])


# Transform to Categorical
df.Sex = pd.Categorical(df.Sex).codes
df.Embarked = pd.Categorical(df.Embarked).codes


print(df.head())


# Normalize - Age, Fare,
normalize_cols = [
    'Age',
    'Pclass',
    'SibSp',
    'Parch',
    'Fare',
]
mean = df[normalize_cols].mean()
std = df[normalize_cols].std()
df[normalize_cols] = (df[normalize_cols] - mean) / std

df = df[['Age', 'Pclass', 'Sex', 'PassengerId', 'Survived']]
df = df[pd.notnull(df['Age'])]
#
# # Shuffle Data!
#
#
# # Split Data: Train, Validation, and Test Sets
# train_size = 0.7
# valid_size = 0.2
# test_size = 0.0
#
# m = df.shape[0]
# prev = 0
# train_data = df[prev:int(m * train_size)]
# prev += int(m * train_size)
# valid_data = df[prev:prev+int(m * valid_size)]
# prev += int(m * valid_size)
# test_data = df[prev:]
#
# train_data.to_pickle('./data/train.pickle')
# valid_data.to_pickle('./data/valid.pickle')
# test_data.to_pickle('./data/test.pickle')
