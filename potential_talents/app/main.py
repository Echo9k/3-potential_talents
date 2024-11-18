import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.features.build_features import *
from src.models.train_model import *

pd.set_option('display.max_colwidth', 200)

# Some variables and functions
search_column_raw = 'job_title'
search_column = 'job_title_cleaned'

def encode_and_get_similarity(data, queries, search_column, output_column):
    data = data.copy()
    
    embeddings = {}
    queries_embeddings = []
    
    # Queries
    starred_queries = data[data['starred'] == True][search_column]
    if len(starred_queries) > 0:
        queries += starred_queries
    # similarities = []
    # for query in queries:
    #     print('START: ' + query)
    #     data = encode_and_get_similarity(data, [query], ['job_title_cleaned'], ['starred_similarity'])
    #     similarities.append(data['starred_similarity'])
        
        
    # starred_similarity = np.mean(similarities, axis=0)
    
    # return starred_similarity

    # without replacing the abbreviations with their full meaning, we will get very bad results
    for index, query in enumerate(queries):
        query = replace_abbreviations(query)
        query = clean_sentence(query)
        queries_embeddings.append(get_bert_embeddings([query]))
        
    # queries_embeddings_mean = np.mean(queries_embeddings, axis=0)
    # queries_embeddings_mean = get_bert_embeddings('Aspiring Human Resources Professional')

    sentences = data[search_column].tolist()

    # Encoding
    embeddings_path = project_root + '/data/processed/data_embeddings.pickle'
    embeddings = get_bert_embeddings(sentences, save_path=embeddings_path, load_path=embeddings_path)

    # Cosine Similarity
    cosine_similarities = []
    for index, query_embeddings in enumerate(queries_embeddings):
        cosine_similarities.append(cosine_similarity(
            query_embeddings,
            embeddings
        )[0])
    # data[output_column] = cosine_similarities[0]
    data[output_column] = np.mean(cosine_similarities, axis=0)

    return data

def update_session_dataset(dataset, columns = None):
    if 'dataset' not in st.session_state or columns is None:
        st.session_state['dataset'] = dataset
    else:
        temp_dataset = st.session_state['dataset']
        temp_dataset.loc[:, columns] = dataset[columns]

        st.session_state['dataset'] = temp_dataset

# Load dataset
def load_dataset():
    if 'dataset' not in st.session_state:
        dataset_path = project_root + '/data/processed/data_processed.csv'
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path)
        else:
            raw_dataset = pd.read_csv(project_root + '/data/raw/data.csv')
            dataset = prepare_dataset(raw_dataset, search_column_raw, save_path=dataset_path)
        dataset['similarity'] = 0
        dataset['starred'] = False

        st.session_state["dataset"] = dataset
    else:
        dataset = st.session_state['dataset']

    return dataset
dataset = load_dataset()

# Hide table row index
hide_table_row_index_style = """
    <style>
        thead tr th:first-child {
            display:none
        }
        tbody th {
            display:none
        }
    </style>
"""
st.markdown(hide_table_row_index_style, unsafe_allow_html=True)

# Set web app title
st.title('Apziva - Potentiel Talents')

# Sidebar
## Github
st.sidebar.subheader('Github')
st.sidebar.write("[readme.md](https://github.com/Ahmant/apziva-potential-talents/tree/main#readme)")
## Abbreviations to replace
st.sidebar.subheader('Abbreviations to replace')
st.sidebar.write('They are automatically replaced')
abbreviations_table_data = pd.DataFrame(columns=('key', 'value'))
for key, value in abbreviations_to_replace.items():
    abbreviations_table_data.loc[len(abbreviations_table_data)] = {'key': key, 'value': value}
st.sidebar.table(abbreviations_table_data)

# Add search input
query = st.text_input('Search query', 'HR Specialist', placeholder='Search...')

# Submit search button
submitted = st.button("Submit")
if submitted:
    with st.spinner('Getting your results...'):
        # Get search results
        dataset = encode_and_get_similarity(dataset, [query], search_column, 'similarity')
        update_session_dataset(dataset)
st.write('"Star (checkbox)" candidates and re-submit to re-rank based')

dataset = dataset.sort_values(['similarity', 'starred'], ascending=False)
update_session_dataset(
    st.data_editor(
        dataset[['id', 'job_title', 'similarity', 'starred']],
        disabled=('id', 'job_title', 'similarity'),
        hide_index=True,
        key="editor",
    ),
    columns=['id', 'job_title', 'similarity', 'starred']
)