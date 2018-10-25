import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stop_words(input_col):
    stop_words = set(stopwords.words("english"))
    
    def remove(sentence):
        if not sentence:
            return None
        words = word_tokenize(sentence)
        meaningful_words = [str(w)\
            if w not in stop_words else ''\
            for w in words]
        return ' '.join(meaningful_words)
    
    return input_col.map(remove)

def get_processed_data():
    file_path = './data/nyt-ingredients-snapshot-2015.csv'
    df_original = pd.read_csv(file_path)

    # Drop Bad Data
    df_original = df_original[df_original.unit != 'crushed']
    df_original = df_original[df_original.input.notna()]

    # Transform Data
    df_original.loc[df_original.unit == 'small pinch', 'unit'] = 'pinch'

    # Remove stop words from input and comment
    df_original['input'] = remove_stop_words(df_original.input)
    df_original['comment'] = remove_stop_words(df_original.comment)

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