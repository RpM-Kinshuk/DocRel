import os
import time
import torch
import pandas as pd
import numpy as np
# import fasttext.util
import tensorflow as tf
from tqdm.auto import tqdm
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, pipeline


dist = 'euclid_'
sv = True
disable_tqdm = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

