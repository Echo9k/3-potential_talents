# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    #%%
def calculate_keyword_match(df, keyword):
    """Calculates the keyword match using fuzzywuzzy's partial_ratio in a vectorized way.

    Args:
      df: Pandas DataFrame with a 'job_title' column.
      keyword: The keyword to match against.

    Returns:
      A NumPy array containing the partial ratio scores.
    """

    return np.vectorize(lambda x: fuzz.partial_ratio(x.lower(), keyword.lower()))(df['job_title'].values)

def calculate_word_frequencies(column):
    """
    Calculate word frequencies in a pandas column using vectorized operations.

    Parameters:
        column (pd.Series): The column containing text data.

    Returns:
        Counter: A Counter object with word frequencies.
    """
    # Split all text into words using a vectorized join and split approach
    all_words = " ".join(column).split()
    # Count word frequencies
    word_counts = Counter(all_words)
    return word_counts


    # Initialize reusable objects
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()


def clean_text_vectorized(text):
    # Check if text is a valid string; otherwise, return an empty string
    if not isinstance(text, str) or not text.strip():
        return ""

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    mask = np.isin(tokens, list(stop_words), invert=True)
    filtered_tokens = np.array(tokens)[mask]

    # Lemmatize
    lemmatized_tokens = np.vectorize(lemmatizer.lemmatize)(filtered_tokens)

    # Detokenize
    return detokenizer.detokenize(lemmatized_tokens)


def cleaning(df, col=None, inplace=False):
    """
    Cleans the specified column(s) of the DataFrame by applying the clean_text_vectorized function.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str or list): The column name or list of column names to clean.
        inplace (bool): Whether to modify the original DataFrame or return a new one.

    Returns:
        pd.DataFrame or None: If inplace is False, returns a modified DataFrame; otherwise, modifies the input DataFrame in place and returns None.
    """
    if col is None:
        raise ValueError("You must specify at least one column name to clean.")

    # If inplace is False, create a copy of the DataFrame to modify
    if not inplace:
        df = df.copy()

    # Process the column(s)
    if isinstance(col, str):  # Single column case
        df[col] = df[col].apply(lambda x: clean_text_vectorized(x) if pd.notna(x) else x)
    elif isinstance(col, list):  # Multiple columns case
        df[col] = df[col].map(lambda x: clean_text_vectorized(x) if pd.notna(x) else x)
    else:
        raise TypeError("col must be a string or a list of strings.")

    # If inplace is True, the original DataFrame is modified
    if inplace:
        return None
    else:
        return df[col]
