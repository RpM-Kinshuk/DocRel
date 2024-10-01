import os
import re
import json
import random
import numpy as np

# File path for storing the variables
DATA_FILE = 'data_storage.json'

def save_data(data):
    if not os.path.exists(os.path.dirname(DATA_FILE)):
        os.makedirs(os.path.dirname(DATA_FILE))
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def modify_data(field_name, new_value):
    data = load_data()
    data[field_name] = new_value  # Add or modify the field
    save_data(data)
    return True  # Indicate that the field was added or modified

def normalize_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Retain alphanumeric characters and spaces
    return text.lower().strip()

def load_cache():
    cache_file = './cache/models_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            return cache_data
    return {}

def save_cache(abstracts, model='none'):
    try:
        cache_file = './cache/models_cache.json'
        index = random.randint(0, len(abstracts) - 1)
        abstract = abstracts[index]

        cache_data = load_cache()

        # Save the index and abstract for the current model
        cache_data[model] = {
            'index': index,
            'abstract': normalize_text(abstract)
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=4)
        return True
    except Exception as e:
        print(e)
        return False

def save_embeddings(embeddings, model='none'):
    np.save(os.path.join('./cache', f'{model}_embeddings.npy'), embeddings)

def load_embeddings(model='none'):
    filename = f'./cache/{model}_embeddings.npy'
    if os.path.exists(filename):
        return np.load(filename)
    return None

def check_embeddings(model, abstracts):
    cached_data = load_cache().get(model, None)
    # tmp_file = os.path.join(os.getcwd(), 'cache', f'{model}_embeddings.npy')
    tmp_file = f'./cache/{model}_embeddings.npy'

    if cached_data and os.path.exists(tmp_file):
        saved_index = cached_data['index']
        saved_abstract = cached_data['abstract']
        
        current_abstract = normalize_text(abstracts[saved_index])

        if current_abstract == saved_abstract:
            print(f'\nUsing cached {model} embeddings\n')
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