import numpy as np
import pickle
import os
import re
import tensorflow as tf
import spacy
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, TFBertModel



# Spacy & NLTK
spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp.pipe_names
stemmer = PorterStemmer()

# BERT Tekenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

abbreviations_to_replace = {
    'GPHR': 'Global Professional in Human Resources',
    'CSR': 'Corporate Social Responsibility',
    'MES': 'Manufacturing Execution Systems',
    'SPHR': 'Senior Professional in Human Resources',
    'SVP': 'Senior Vice President',
    'GIS': 'Geographic Information System',
    'RRP': 'Reduced Risk Products',
    'CHRO': 'Chief Human Resources Officer',
    'HRIS': 'Human resources information system',
    'HR': 'Human resources',
}

def replace_abbreviations(sentence):
    replaced_sentence = sentence
    for abbreviation, replacement in abbreviations_to_replace.items():
        # Create a regular expression pattern to match the whole word
        pattern = r'\b{}\b'.format(re.escape(abbreviation))
    
        # Use re.sub() to replace the word in the sentence
        replaced_sentence = re.sub(pattern, replacement, replaced_sentence, flags=re.IGNORECASE)

    return replaced_sentence

def clean_sentence(sentence):
    # Remove special characters
    new_sentence = re.sub(r'[+*,.|(){}&\-\']', '', sentence)

    # Replce abbreviations
    new_sentence = replace_abbreviations(new_sentence)
    
    words = new_sentence.split()
    
    # Stemming
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
        
    # Lemmatization
    lemmatized_words = []
    doc = spacy_nlp(" ".join(stemmed_words))
    for token in doc:
        if not token.is_stop:
            lemmatized_words.append(token.lemma_)

    return " ".join(lemmatized_words)

def get_bert_embeddings(sentences, save_path=None, load_path=None):
    embeddings_dict = {}
    embeddings = []

    if load_path is not None and os.path.exists(load_path):
        with open(load_path, 'rb') as pkl:
            embeddings_dict = pickle.load(pkl)

    for sentence in sentences:
        if sentence not in embeddings_dict:
            # Tokenize input sentence
            encoded_inputs = bert_tokenizer(sentence, padding=True, truncation=True, return_tensors='tf')
        
            # Generate BERT embeddings
            outputs = bert_model(encoded_inputs)
            hidden_states = outputs.last_hidden_state

            # Apply pooling strategy - averaging
            pooled = tf.reduce_mean(hidden_states, axis=1)
            pooled_embedding = pooled.numpy().reshape(-1)
            embeddings.append(pooled_embedding)
            embeddings_dict[sentence] = pooled_embedding
        else:
            embeddings.append(embeddings_dict[sentence])
    
    if save_path is not None:
        with open(save_path, 'wb') as pkl:
            pickle.dump(embeddings_dict, pkl)


    return np.array(embeddings)


def calculate_keyword_match(df, keyword):
    """Calculates the keyword match using fuzzywuzzy's partial_ratio in a vectorized way.

    Args:
      df: Pandas DataFrame with a 'job_title' column.
      keyword: The keyword to match against.

    Returns:
      A NumPy array containing the partial ratio scores.
    """

    return np.vectorize(lambda x: fuzz.partial_ratio(x.lower(), keyword.lower()))(df['job_title'].values)


def calculate_keyword_match(df, search_phrases):
    """Calculates the keyword match using fuzzywuzzy's partial_ratio for multiple phrases.

    Args:
    df: Pandas DataFrame with a 'job_title' column.
    search_phrases: A list of keywords to match against.

    Returns:
    A NumPy array containing the maximum partial ratio score for each row.
    """
    if isinstance(search_phrases, str):
        return np.vectorize(lambda x: fuzz.partial_ratio(x.lower(), search_phrases.lower()))
        (df['job_title'].values)

    elif isinstance(search_phrases, list):
        # Find the maximum partial ratio score for each row across all search phrases
        return np.max([
            np.vectorize(lambda x: fuzz.partial_ratio(x.lower(), phrase.lower()))(
                df['job_title'].values) for phrase in search_phrases
        ], axis=0)  # partial_ratios

# Tokenize dataset
def tokenize_function(examples):
    return bert_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)