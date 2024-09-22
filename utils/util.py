import os
import numpy as np

def save_embeddings(embeddings, filename='embeddings.npy'):
    np.save(os.path.join('./cache', filename), embeddings)

def load_embeddings(filename):
    if os.path.exists(filename):
        return np.load(filename)
    return None

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