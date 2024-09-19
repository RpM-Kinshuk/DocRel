import time
import torch
import numpy as np
from tqdm.auto import tqdm

def calculate_similarity_scores(summary_embeddings, goal_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import euclidean_distances
    # print('Euclidean Distance is being used')
    euclid_sim_scores = euclidean_distances(summary_embeddings, goal_embeddings)
    # print('Cosine Similarity is being used')
    cos_sim_scores = cosine_similarity(summary_embeddings, goal_embeddings)
    return cos_sim_scores, euclid_sim_scores

def use(abstracts, query, disable_tqdm=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    import tensorflow_hub as hub
    # Load the USE model
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    # Process summaries in batches
    batch_size = 100
    n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

    cos_sim_scores = []
    euclid_sim_scores = []

    query_embedding = use_model([query]).numpy() # type: ignore
    for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
        batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
        summary_embeddings = use_model(batch_summaries).numpy() # type: ignore
        cos_batch_scores, euclid_batch_scores = calculate_similarity_scores(summary_embeddings, query_embedding)
        cos_sim_scores.extend(cos_batch_scores)
        euclid_sim_scores.extend(euclid_batch_scores)

    return cos_sim_scores, euclid_sim_scores

def stf(abstracts, query, disable_tqdm=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    from sentence_transformers import SentenceTransformer
    # Load the SentenceTransformer model
    stf_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device) # type: ignore
    def get_stf_query_embedding(query):
        query_embedding = stf_model.encode(query, convert_to_tensor=True, device=device) # type: ignore
        return query_embedding.cpu().numpy()
    query_embedding = get_stf_query_embedding(query)

    batch_size = 32
    n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

    cos_sim_scores = []
    euclid_sim_scores = []

    for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
        batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
        summary_embeddings = stf_model.encode(batch_summaries, convert_to_tensor=True, device=device).cpu().numpy() # type: ignore
        cos_batch_scores, euclid_batch_scores = calculate_similarity_scores(summary_embeddings, query_embedding)
        cos_sim_scores.extend(cos_batch_scores)
        euclid_sim_scores.extend(euclid_batch_scores)

    return cos_sim_scores, euclid_sim_scores

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
    batch_size = 100
    n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

    cos_sim_scores = []
    euclid_sim_scores = []

    for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
        batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
        summary_embeddings = np.array([get_fasttext_embedding(summary) for summary in batch_summaries])
        cos_batch_scores, euclid_batch_scores = calculate_similarity_scores(summary_embeddings, query_embedding)
        cos_sim_scores.extend(cos_batch_scores)
        euclid_sim_scores.extend(euclid_batch_scores)

    return cos_sim_scores, euclid_sim_scores

def glove(abstracts, query, disable_tqdm=True):
    # Load GloVe embeddings
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
    batch_size = 100
    n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

    cos_sim_scores = []
    euclid_sim_scores = []

    for i in tqdm(range(n_batches), desc='Processing Batches', disable=disable_tqdm):
        batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
        summary_embeddings = np.array([get_glove_embedding(summary, glove_model) for summary in batch_summaries])
        cos_batch_scores, euclid_batch_scores = calculate_similarity_scores(summary_embeddings, query_embedding)
        cos_sim_scores.extend(cos_batch_scores)
        euclid_sim_scores.extend(euclid_batch_scores)

    return cos_sim_scores, euclid_sim_scores

def elmo(abstracts, query, disable_tqdm=True):
    import tensorflow as tf
    import tensorflow_hub as hub
    # Load ELMo model
    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
    def get_elmo_embedding(texts):
        # embeddings = elmo_model.signatures['default'](tf.constant(texts))['elmo']
        # return tf.reduce_mean(embeddings, axis=1).numpy()
        output_dict = elmo_model.signatures['default'](tf.constant(texts)) # type: ignore
        return output_dict['default'].numpy()
    query_embedding = get_elmo_embedding([query])

    batch_size = 100
    n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

    cos_sim_scores = []
    euclid_sim_scores = []

    for i in tqdm(range(n_batches), desc="Processing Summaries"):
        batch_summaries = abstracts[i * batch_size:(i + 1) * batch_size]
        summary_embeddings = get_elmo_embedding(batch_summaries)
        cos_batch_scores, euclid_batch_scores = calculate_similarity_scores(summary_embeddings, query_embedding)
        cos_sim_scores.extend(cos_batch_scores)
        euclid_sim_scores.extend(euclid_batch_scores)

    return cos_sim_scores, euclid_sim_scores