import matplotlib.pyplot as plt
import pandas as pd;

df = pd.read_csv('./data/iris.csv')

print(df.head())

xlabel = 'SepalLengthCm'
ylabel = 'SepalWidthCm'

plt.scatter(df[xlabel], df[ylabel])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()

# SepalLengthCm  SepalWidthCm

print('\tMin\tMax')
print('SepLen\t{}\t{}'.format(df['SepalLengthCm'].min(), df['SepalLengthCm'].max()))
print('SepWid\t{}\t{}'.format(df['SepalWidthCm'].min(), df['SepalWidthCm'].max()))
