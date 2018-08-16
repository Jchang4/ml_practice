import pandas as pd

df = pd.read_csv('./data/submission.csv')

df.Survived = pd.Categorical(df.Survived).codes


print(df.head())

df.to_csv('./data/submission.csv', index=False, index_label=False)
