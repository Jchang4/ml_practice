import matplotlib.pyplot as plt
import pandas as pd;

df = pd.read_csv('./data/iris.csv')
df.Species = pd.Categorical(df.Species)
df['Species_codes'] = df.Species.cat.codes

print(df.head())

plt.scatter('PetalLengthCm', 'PetalWidthCm', c='Species_codes', data=df)
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
plt.show()

# SepalLengthCm  SepalWidthCm

print('\tMin\tMax')
print('SepLen\t{}\t{}'.format(df['SepalLengthCm'].min(), df['SepalLengthCm'].max()))
print('SepWid\t{}\t{}'.format(df['SepalWidthCm'].min(), df['SepalWidthCm'].max()))
