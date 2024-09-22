import os
import json
import numpy as np

# File path for storing the variables
DATA_FILE = 'data_storage.json'

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(index, abstract, filename='none_cache.json'):
    cache_data = {
        'index': index,
        'abstract': abstract
    }
    with open(filename, 'w') as f:
        json.dump(cache_data, f)

def load_cache(filename='none_cache.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def save_embeddings(embeddings, filename='embeddings.npy'):
    np.save(os.path.join('./cache', filename), embeddings)

def load_embeddings(filename):
    if os.path.exists(filename):
        return np.load(filename)
    return None

def check_embeddings(model, abstracts):
    if model is None:
        return False
    

def calculate_similarity_scores(abstract_embeddings, goal_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    euclid_sim_scores = euclidean_distances(abstract_embeddings, goal_embeddings)
    cos_sim_scores = cosine_similarity(abstract_embeddings, goal_embeddings)
    return cos_sim_scores, euclid_sim_scores

def analyze_lengths(dataset):
    print('Word-wise')
    dataset['Abstract Length in Words'] = dataset['abstract'].apply(lambda x: len(str(x).split()))
    print(dataset['Abstract Length in Words'].describe())

    print('\nCharacter-wise')
    dataset['Abstract Length in Characters'] = dataset['abstract'].apply(lambda x: len(str(x)))
    print(dataset['Abstract Length in Characters'].describe())