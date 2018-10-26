from helpers.model import load_data
import helpers.data_viz as data_viz


df = load_data()

print(df.describe())
