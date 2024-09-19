import os
import time
import torch
import numpy as np
from util import *
import pandas as pd
# import fasttext.util
import tensorflow as tf
from tqdm.auto import tqdm
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


dist = 'euclid_'
sv = True
disable_tqdm = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def similarity_scores(model, query):
    dataset = pd.read_csv('./scopus_results.csv')
    # dataset = dataset[:1000]
    abstracts = dataset['abstract'].tolist()

def analyze_lengths(dataset):
    print('Word-wise')
    dataset['Abstract Length in Words'] = dataset['abstract'].apply(lambda x: len(str(x).split()))
    print(dataset['Abstract Length in Words'].describe())

    print('\nCharacter-wise')
    dataset['Abstract Length in Characters'] = dataset['abstract'].apply(lambda x: len(str(x)))
    print(dataset['Abstract Length in Characters'].describe())