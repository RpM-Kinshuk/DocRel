import os
import json
import random
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

def save_cache(abstracts, model='none'):
    try:
        cache_file = './cache/models_cache.json'
        index = random.randint(0, len(abstracts) - 1)
        abstract = abstracts[index]

        # Load existing cache or initialize a new one
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
        else:
            cache_data = {}

        # Save the index and abstract for the current model
        cache_data[model] = {
            'index': index,
            'abstract': abstract
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(e)

def load_cache(model='none'):
    cache_file = './cache/models_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            return cache_data.get(model, None)
    return None

def save_embeddings(embeddings, model='none'):
    np.save(os.path.join('./cache', f'{model}_embeddings.npy'), embeddings)

def load_embeddings(model='none'):
    filename = f'./cache/{model}_embeddings.npy'
    if os.path.exists(filename):
        return np.load(filename)
    return None

def check_embeddings(model, abstracts):
    cached_data = load_cache(model)
    tmp_file = f'./cache/{model}_embeddings.npy'
    if cached_data and os.path.exists(tmp_file):
        saved_index = cached_data['index']
        saved_abstract = cached_data['abstract']
        
        current_abstract = abstracts[saved_index]
        if current_abstract == saved_abstract:
            return True
        else:
            return False
    return False

def calculate_similarity_scores(abstract_embeddings, goal_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    if goal_embeddings.ndim == 1:
        goal_embeddings = goal_embeddings.reshape(1, -1)
    if abstract_embeddings.ndim == 1:
        abstract_embeddings = abstract_embeddings.reshape(1, -1)
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