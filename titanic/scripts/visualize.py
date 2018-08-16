import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./data/train.csv')

y = df['Pclass']
x = df['Age']
c = df['Survived']

plt.scatter(x, y, c=c)
plt.xlabel('Age')
plt.ylabel('Pclass')
plt.show()
