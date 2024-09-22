import torch
import numpy as np
from tqdm.auto import tqdm
from utils.util import save_cache
from utils.util import save_embeddings
from utils.util import load_embeddings
from utils.util import check_embeddings

def use(abstracts, query, disable_tqdm=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    import tensorflow_hub as hub
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    query_embedding = use_model([query]).numpy() # type: ignore
    abstract_embeddings = []

    if check_embeddings('use', abstracts):
        abstract_embeddings = load_embeddings('use_embeddings.npy')
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

        for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
            batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
            batch_embeddings = use_model(batch_summaries).numpy() # type: ignore
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, 'use_embeddings.npy')
        save_cache(abstracts, 'use')

    return abstract_embeddings, query_embedding

def stf(abstracts, query, disable_tqdm=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    from sentence_transformers import SentenceTransformer
    stf_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device) # type: ignore
    def get_stf_query_embedding(query):
        query_embedding = stf_model.encode(query, convert_to_tensor=True, device=device) # type: ignore
        return query_embedding.cpu().numpy()
    
    query_embedding = get_stf_query_embedding(query)
    abstract_embeddings = []

    if check_embeddings('stf', abstracts):
        abstract_embeddings = load_embeddings('stf_embeddings.npy')
    else:
        batch_size = 32
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
            batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
            batch_embeddings = stf_model.encode(batch_summaries, convert_to_tensor=True, device=device).cpu().numpy() # type: ignore
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, 'stf_embeddings.npy')
        save_cache(abstracts, 'stf')
    
    return abstract_embeddings, query_embedding

def fasttext(abstracts, query, disable_tqdm=True):
    # import fasttext
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    fasttext_model = fasttext.load_model('./cc.en.300.bin')
    def get_fasttext_embedding(text):
        global not_identified, count_not_indent, total_number
        words = text.split()
        word_vecs = []
        count = 0
        for word in words:
            if word in fasttext_model:
                word_vecs.append(fasttext_model[word])
        # word_vecs = [fasttext_model.get_word_vector(word) for word in words if word in fasttext_model]
        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(300)
    
    query_embedding = get_fasttext_embedding(query)
    abstract_embeddings = []

    if check_embeddings('fasttext', abstracts):
        abstract_embeddings = load_embeddings('fasttext_embeddings.npy')
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
            batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
            batch_embeddings = np.array([get_fasttext_embedding(summary) for summary in batch_summaries])
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, 'fasttext_embeddings.npy')
        save_cache(abstracts, 'fasttext')

    return abstract_embeddings, query_embedding

def glove(abstracts, query, disable_tqdm=True):
    def load_glove_model(glove_file):
        print("Loading GloVe Model")
        glove_model = {}
        with open(glove_file, 'r', encoding='utf8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print("Done.", len(glove_model), " words loaded!")
        return glove_model
    glove_model = load_glove_model('./glove.6B.300d.txt')
    def get_glove_embedding(text, glove_model, embedding_dim=300):
        global not_identified, count_not_indent, total_number
        words = text.split()
        word_vecs = []
        for word in words:
            if word in glove_model:
                word_vecs.append(glove_model[word])
        # word_vecs = [glove_model[word] for word in words if word in glove_model]
        if word_vecs:
            return np.mean(word_vecs, axis=0)
        else:
            return np.zeros(embedding_dim)
    
    query_embedding = get_glove_embedding(query, glove_model)
    abstract_embeddings = []

    if check_embeddings('glove', abstracts):
        abstract_embeddings = load_embeddings('glove_embeddings.npy')
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
            batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
            batch_embeddings = np.array([get_glove_embedding(summary, glove_model) for summary in batch_summaries])
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, 'glove_embeddings.npy')
        save_cache(abstracts, 'glove')

    return abstract_embeddings, query_embedding

def elmo(abstracts, query, disable_tqdm=True):
    import tensorflow as tf
    import tensorflow_hub as hub
    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    def get_elmo_embedding(texts):
        # embeddings = elmo_model.signatures['default'](tf.constant(texts))['elmo']
        # return tf.reduce_mean(embeddings, axis=1).numpy()
        output_dict = elmo_model.signatures['default'](tf.constant(texts)) # type: ignore
        return output_dict['default'].numpy()
    
    query_embedding = get_elmo_embedding([query])
    abstract_embeddings = []

    if check_embeddings('elmo', abstracts):
        abstract_embeddings = load_embeddings('elmo_embeddings.npy')
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(range(n_batches), desc="Processing Summaries"):
            batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
            batch_embeddings = get_elmo_embedding(batch_summaries)
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, 'elmo_embeddings.npy')
        save_cache(abstracts, 'elmo')
        
    return abstract_embeddings, query_embedding