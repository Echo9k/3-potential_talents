# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import numpy as np
from collections import Counter
# Geolocation
from geopy.geocoders import Nominatim
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the geolocator with a custom user-agent
geolocator = Nominatim(user_agent="geoapiExercises")

# Use RateLimiter to respect the service's rate limits
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Initialize a cache dictionary
location_cache = {}


# Initialize reusable objects
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
detokenizer = TreebankWordDetokenizer()


def standardize_location(location):
    """
    Standardizes location names to "City, Country" format.
    """
    if pd.isnull(location) or location.strip() == '':
        logger.warning("Empty or null location provided.")
        return np.nan

    # Check if the location is already cached
    if location in location_cache:
        return location_cache[location]

    try:
        # Geocode the location
        loc = geocode(location, language='en', addressdetails=True)
        if loc and 'address' in loc.raw:
            address = loc.raw['address']
            # Extract city
            city = address.get('city') or address.get('town') or address.get('village') \
                   or address.get('hamlet') or address.get('municipality') or address.get('county')
            # Extract country
            country = address.get('country')
            if city and country:
                standardized_loc = f"{city}, {country}"
            elif country:
                standardized_loc = country
            else:
                standardized_loc = loc.address  # Fallback to full address
            # Cache the result
            location_cache[location] = standardized_loc
            return standardized_loc
        else:
            logger.warning(f"Could not geocode location: '{location}'")
            location_cache[location] = location  # Cache the original location
            return location
    except Exception as e:
        logger.error(f"Error geocoding location '{location}': {e}")
        location_cache[location] = location  # Cache the original location
        return location

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