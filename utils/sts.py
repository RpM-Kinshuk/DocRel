import pandas as pd
from utils.embed import *
import matplotlib.pyplot as plt
from utils.util import calculate_overlap
from utils.util import calculate_similarity_scores

sv = False
disable_tqdm = True

def save_scores(dataset, cos_sim_scores, euclid_sim_scores, keyword_overlaps, model, sim_measure, top_n):
    cos_scores_df = pd.DataFrame(cos_sim_scores, columns=[f'Similarity'])
    euclid_scores_df = pd.DataFrame(euclid_sim_scores, columns=[f'Similarity'])
    overlap_df = pd.DataFrame(keyword_overlaps, columns=[f'Keyword Overlap'])
    cos_result_df = pd.concat([dataset, cos_scores_df, overlap_df], axis=1)
    euclid_result_df = pd.concat([dataset, euclid_scores_df, overlap_df], axis=1)
    if sv:
        cos_result_df.to_csv(f'./cache/{model}_cosine_scores.csv', index=False)
        euclid_result_df.to_csv(f'./cache/{model}_euclid_scores.csv', index=False)
    
    current_df = cos_result_df if sim_measure == 'cosine' else euclid_result_df
    top_results = current_df.sort_values(by='Similarity', ascending=False if sim_measure == 'cosine' else True).head(int(top_n))
    return top_results

def similarity_scores(query, top_n, model, sim_measure):
    dataset = pd.read_csv('./scopus_results.csv')
    if 'abstract' in dataset.columns:
        abstracts = dataset['abstract'].astype(str).tolist()
    elif 'Abstract' in dataset.columns:
        abstracts = dataset['Abstract'].astype(str).tolist()
    else:
        raise ValueError('No abstract column found in dataset')
    abstract_embeddings, query_embedding = None, None
    if model == 'use':
        abstract_embeddings, query_embedding = use(abstracts, query, disable_tqdm)
    elif model == 'stf':
        abstract_embeddings, query_embedding = stf(abstracts, query, disable_tqdm)
    elif model == 'fasttext':
        abstract_embeddings, query_embedding = fstext(abstracts, query, disable_tqdm)
    elif model == 'glove':
        abstract_embeddings, query_embedding = glove(abstracts, query, disable_tqdm)
    elif model == 'elmo':
        abstract_embeddings, query_embedding = elmo(abstracts, query, disable_tqdm)
    else:
        raise ValueError('Invalid model selected')
    
    cos_sim_scores, euclid_sim_scores = calculate_similarity_scores(abstract_embeddings, query_embedding)
    keyword_overlaps = calculate_overlap(abstracts, query)

    top_results = save_scores(dataset, cos_sim_scores, euclid_sim_scores, keyword_overlaps, model, sim_measure, top_n)
    return top_results