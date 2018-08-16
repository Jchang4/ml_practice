import pandas as pd

df = pd.read_csv('./data/iris.csv')

m = df.shape[0]
n = df.shape[1]-1
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

prev = 0
train_size = 0.6
valid_size = 0.2
test_size = 0.2

# normalize each feature
mean = df[columns].mean() #.values.reshape(1, len(columns))
sigma = df[columns].std() #.values.reshape(1, len(columns))

df[columns] = (df[columns] - mean) / sigma

# Turn labels into numbers
df['Species'] = pd.Categorical(df.Species).codes





# ======================================================================
# Make Binary
# df = df.loc[(df['Species'] == 0) | (df['Species'] == 1)]
# ======================================================================



# Randomize
df = df.sample(frac=1).reset_index(drop=True)

# Split data
train_data = df[prev:int(m * train_size)]
prev += int(m * train_size)

valid_data = df[prev:prev+int(m * valid_size)]
prev += int(m  * valid_size)

test_data = df[prev:]

# Save data
train_data.to_pickle('./data/train.pickle')
valid_data.to_pickle('./data/valid.pickle')
test_data.to_pickle('./data/test.pickle')
