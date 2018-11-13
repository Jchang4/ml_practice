import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def replace_fractions_with_decimals(s):
    def add_nums(m):
        return str(float(int(m.group(1)) + float(m.group(2)) / float(m.group(3))))
    new_s = re.sub(r'(\d+)\s+(\d+)/(\d+)', add_nums, s)
    new_s = re.sub(r'(\d+)/(\d+)', lambda m: str(float(m.group(1)) / float(m.group(2))), new_s)
    return new_s

def remove_stop_words(sentence):
    words = word_tokenize(sentence)
    meaningful_words = [str(w)\
        if w not in stop_words else ''\
        for w in words]
    return ' '.join(meaningful_words)

def get_processed_data():
    """
        * Remove stop words from input and columns
            - rename 'input' to 'original_input'
            - 'input' is now the processed input
        * 
    """
    file_path = './data/nyt-ingredients-snapshot-2015.csv'
    df_original = pd.read_csv(file_path)

    # Drop Bad Data
    df_original = df_original[df_original.unit != 'crushed']
    df_original = df_original[df_original.input.notna()]

    # Transform Data
    # df_original.loc[df_original.unit == 'small pinch', 'unit'] = 'pinch'

    # Remove stop words from input and comment
    df_original['input'] = df_original.input.map(remove_stop_words, na_action='ignore')
    df_original['comment'] = df_original.comment.map(remove_stop_words, na_action='ignore')

    # Replace Fractions with Decimals
    df_original['input'] = df_original.input.map(replace_fractions_with_decimals)
    df_original['comment'] = df_original.comment.map(replace_fractions_with_decimals, na_action='ignore')

    # Lemmatize name and units

    # Rename Columns
    df_new = pd.DataFrame()
    df_new['original_input'] = df_original.input
    df_new['input'] = df_original.input.str.lower()
    df_new['name'] = df_original.name.str.lower()
    df_new['quantity'] = df_original.qty
    df_new['unit'] = df_original.unit.str.lower()
    df_new['comment'] = df_original.comment.str.lower()

    return df_new


if __name__ == '__main__':
    df_processed = get_processed_data()
    df_processed.to_pickle('./data/nyt-data.pickle')

    # df = pd.read_csv('./data/nyt-ingredients-snapshot-2015.csv')
