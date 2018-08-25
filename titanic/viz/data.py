import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/train.csv')

survived = df.Survived


print(df.head())
# print(df.columns)
# print(df.groupby('Pclass').count())


[
'Pclass',
'Sex',
'Age',
'SibSp',
'Parch',
'Ticket',
'Fare',
'Cabin',
'Embarked'
]

idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
bar_width = 0.3

# plt.subplot(211)
plt.bar(df[survived == 1].Age.unique(),
        df[survived == 1].groupby('Age').count().Survived,
        color='b', label='Survived', width=bar_width)
plt.bar(df[survived == 0].Age.unique() + bar_width,
        df[survived == 0].groupby('Age').count().Survived,
        color='r', label='Died', width=bar_width)
plt.show()
