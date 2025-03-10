import os
import torch
import logging
import fasttext
import numpy as np
from tqdm.auto import tqdm
from utils.util import save_cache
from utils.util import save_embeddings
from utils.util import load_embeddings
from utils.util import check_embeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

tqdm_off = False
cache_dir = f"{os.getcwd()}/cache/models"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


def log_gpu_info(name):
    """Log GPU information for monitoring."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2  # MB

        logger.info(f"===== {name} GPU INFO =====")
        logger.info(f"GPU Available: True - Using {device_name}")
        logger.info(f"Device Count: {device_count}")
        logger.info(f"Current Device: {current_device}")
        logger.info(f"Memory Allocated: {memory_allocated:.2f} MB")
        logger.info(f"Memory Reserved: {memory_reserved:.2f} MB")
    else:
        logger.info(f"===== {name} GPU INFO =====")
        logger.info("GPU not available, using CPU")


def use(
    abstracts,
    query,
    disable_tqdm=tqdm_off,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    import tensorflow_hub as hub

    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    query_embedding = use_model([query]).numpy()  # type: ignore
    abstract_embeddings = []

    if check_embeddings("use", abstracts):
        abstract_embeddings = load_embeddings("use")
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division

        for i in tqdm(
            range(n_batches), desc="Processing Batches", disable=disable_tqdm
        ):
            batch_summaries = abstracts[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = use_model(batch_summaries).numpy()  # type: ignore
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, "use")
        save_cache(abstracts, "use")

    # remove model from memory
    del use_model
    torch.cuda.empty_cache()
    return abstract_embeddings, query_embedding


def stf(
    abstracts,
    query,
    disable_tqdm=tqdm_off,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    from sentence_transformers import SentenceTransformer

    stf_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device, cache_folder=cache_dir)  # type: ignore

    log_gpu_info("STF - After Model Load")

    def get_stf_query_embedding(query):
        logger.info(f"Encoding query on device: {device}")
        query_embedding = stf_model.encode(query, convert_to_tensor=True, device=device)  # type: ignore
        return query_embedding.cpu().numpy()

    query_embedding = get_stf_query_embedding(query)
    abstract_embeddings = []

    if check_embeddings("stf", abstracts):
        logger.info("Loading cached STF embeddings")
        abstract_embeddings = load_embeddings("stf")
    else:
        logger.info(f"Computing STF embeddings for {len(abstracts)} abstracts")
        batch_size = 32
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(
            range(n_batches), desc="Processing Batches", disable=disable_tqdm
        ):
            batch_summaries = abstracts[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = (
                stf_model.encode(
                    batch_summaries,
                    convert_to_tensor=True,
                    device=device,  # type: ignore
                )
                .cpu()
                .numpy()
            )
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, "stf")
        save_cache(abstracts, "stf")

    log_gpu_info("STF - Before Cleanup")

    stf_model.to("cpu")
    del stf_model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    return abstract_embeddings, query_embedding


def fstext(abstracts, query, disable_tqdm=tqdm_off):
    # import fasttext
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    fasttext_model = fasttext.load_model("cc.en.300.bin")

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

    if check_embeddings("fasttext", abstracts):
        abstract_embeddings = load_embeddings("fasttext")
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(
            range(n_batches), desc="Processing Batches", disable=disable_tqdm
        ):
            batch_summaries = abstracts[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = np.array(
                [get_fasttext_embedding(summary) for summary in batch_summaries]
            )
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, "fasttext")
        save_cache(abstracts, "fasttext")

    del fasttext_model
    torch.cuda.empty_cache()
    return abstract_embeddings, query_embedding


def glove(abstracts, query, disable_tqdm=tqdm_off):
    def load_glove_model(glove_file):
        print("Loading GloVe Model")
        glove_model = {}
        with open(glove_file, "r", encoding="utf8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array(split_line[1:], dtype=np.float64)
                glove_model[word] = embedding
        print("Done.", len(glove_model), " words loaded!")
        return glove_model

    glove_model = load_glove_model("./glove.6B.300d.txt")

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

    if check_embeddings("glove", abstracts):
        abstract_embeddings = load_embeddings("glove")
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(
            range(n_batches), desc="Processing Batches", disable=disable_tqdm
        ):
            batch_summaries = abstracts[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = np.array(
                [
                    get_glove_embedding(summary, glove_model)
                    for summary in batch_summaries
                ]
            )
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, "glove")
        save_cache(abstracts, "glove")

    del glove_model
    torch.cuda.empty_cache()
    return abstract_embeddings, query_embedding


def elmo(abstracts, query, disable_tqdm=tqdm_off):
    import tensorflow as tf
    import tensorflow_hub as hub

    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    elmo_model = hub.load("https://tfhub.dev/google/elmo/3")

    def get_elmo_embedding(texts):
        # embeddings = elmo_model.signatures['default'](tf.constant(texts))['elmo']
        # return tf.reduce_mean(embeddings, axis=1).numpy()
        output_dict = elmo_model.signatures["default"](tf.constant(texts))  # type: ignore
        return output_dict["default"].numpy()

    query_embedding = get_elmo_embedding([query])
    abstract_embeddings = []

    if check_embeddings("elmo", abstracts):
        abstract_embeddings = load_embeddings("elmo")
    else:
        batch_size = 100
        n_batches = (len(abstracts) + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(
            range(n_batches), desc="Processing Summaries", disable=disable_tqdm
        ):
            batch_summaries = abstracts[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = get_elmo_embedding(batch_summaries)
            abstract_embeddings.append(batch_embeddings)

        abstract_embeddings = np.vstack(abstract_embeddings)
        save_embeddings(abstract_embeddings, "elmo")
        save_cache(abstracts, "elmo")

    del elmo_model
    torch.cuda.empty_cache()
    return abstract_embeddings, query_embedding
