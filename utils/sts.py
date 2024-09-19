import os
import pandas as pd
from utils.util import *
import matplotlib.pyplot as plt

sv = True
disable_tqdm = True

def similarity_scores(query, top_n, model, sim_measure):
    dataset = pd.read_csv('./scopus_results.csv')
    if os.path.exists(f'./{model}_cosine_scores.csv'):
        tmp_df = pd.read_csv(f'./{model}_cosine_scores.csv')
        tmp_df.drop(columns=['Similarity'], inplace=True)
        if tmp_df.equals(dataset):
            tmp_str = 'cosine' if sim_measure == 'cosine' else 'euclid'
            current_df = pd.read_csv(f'./{model}_{tmp_str}_scores.csv')
            top_results = current_df.sort_values(by='Similarity', ascending=False if sim_measure == 'cosine' else True).head(int(top_n))
            return top_results
    # dataset = dataset[:1000]
    # Make a list of str out of the abstracts
    abstracts = dataset['abstract'].astype(str).tolist()
    if model == 'use':
        cos_sim_scores, euclid_sim_scores = use(abstracts, query, disable_tqdm)
    elif model == 'stf':
        cos_sim_scores, euclid_sim_scores = stf(abstracts, query, disable_tqdm)
    elif model == 'fasttext':
        cos_sim_scores, euclid_sim_scores = fasttext(abstracts, query, disable_tqdm)
    elif model == 'elmo':
        cos_sim_scores, euclid_sim_scores = elmo(abstracts, query, disable_tqdm)
    else:
        raise ValueError('Invalid model selected')
    
    cos_scores_df = pd.DataFrame(cos_sim_scores, columns=[f'Similarity'])
    euclid_scores_df = pd.DataFrame(euclid_sim_scores, columns=[f'Similarity'])
    cos_result_df = pd.concat([dataset, cos_scores_df], axis=1)
    euclid_result_df = pd.concat([dataset, euclid_scores_df], axis=1)
    if sv:
        cos_result_df.to_csv(f'./{model}_cosine_scores.csv', index=False)
        euclid_result_df.to_csv(f'./{model}_euclid_scores.csv', index=False)
    
    current_df = cos_result_df if sim_measure == 'cosine' else euclid_result_df
    top_results = current_df.sort_values(by='Similarity', ascending=False if sim_measure == 'cosine' else True).head(int(top_n))
    return top_results

def analyze_lengths(dataset):
    print('Word-wise')
    dataset['Abstract Length in Words'] = dataset['abstract'].apply(lambda x: len(str(x).split()))
    print(dataset['Abstract Length in Words'].describe())

    print('\nCharacter-wise')
    dataset['Abstract Length in Characters'] = dataset['abstract'].apply(lambda x: len(str(x)))
    print(dataset['Abstract Length in Characters'].describe())