import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('./data/iris.csv')
df = df.sample(frac=1).reset_index(drop=True) # shuffle df
df['Species'] = pd.Categorical(df['Species'])


X = df[['SepalLengthCm', 'SepalWidthCm']]  # we only take the first two features.
Y = df['Species']

h = .02  # step size in the mesh

logreg = LogisticRegression()

logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X['SepalLengthCm'].min() - .5, X['SepalLengthCm'].max() + .5
y_min, y_max = X['SepalWidthCm'].min() - .5, X['SepalWidthCm'].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X['SepalLengthCm'], X['SepalWidthCm'], c=df.Species.cat.codes, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# 'PetalLengthCm', 'PetalWidthCm'
