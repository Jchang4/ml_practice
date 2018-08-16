import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('./data/train.csv')

survived = df[df['Survived'] == 1]
died = df[df['Survived'] == 0]

x1 = survived['Embarked']
y1 = survived['Age']

x2 = died['Embarked']
y2 = died['Age']


plt.subplot(211)
plt.plot(x1, y1, 'bo',
        x2, y2, 'rx')
plt.xlabel('Count')
plt.ylabel('Age')
plt.show()


print(df.info())
print(df.groupby(['Embarked']))
