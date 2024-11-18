import os
import sys
from src.features.build_features import *

def prepare_dataset(dataset, column, save_path):
    dataset = dataset.copy()

    column_cleaned = column + '_cleaned'

    # Clean sentences
    dataset[column_cleaned] = dataset[column].apply(clean_sentence)

    # Get embedings
    # dataset[column + '_embeddings'] = get_bert_embeddings(dataset[column_cleaned])

    # Save processed/cleaned dataset
    if save_path is not None:
        dataset.to_csv(save_path)
    
    return dataset