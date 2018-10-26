import pandas as pd

def load_data(filePath):
    return pd.read_pickle(filePath or './data/nyt-data.pickle')