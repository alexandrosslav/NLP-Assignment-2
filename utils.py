from pathlib import Path
import urllib.request
import os
import pickle

import clslib

def download_pretrained(option):
    """
    You can use this to download GloVe (or similar) model using wget. 
    It's fine to download it through other means, but don't unzip the file (the rest of the notebook can 
    handle the file in its zipped format).

    This function returns the path which you can use for the loader class that follows.

    The function will skip the download if it detects the file in disk. If the file in disk is corrupted, you will have to delete the file first.
    """
    if option == 'glove':
        destination = "glove.2024.wikigiga.50d.zip"
        dims = 50
        skip_first = False
    elif option == 'fasttext':
        destination = "wiki-news-300d-1M.vec.zip"
        dims = 300
        skip_first = True
    else:
        raise ValueError(f"Unkonwn option: {option}")
        
    if not Path(destination).is_file():
        if option == 'glove':
            # !wget https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip
            url = "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip"
            urllib.request.urlretrieve(url, destination)

        elif option == 'fasttext':
            # !wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
            urllib.request.urlretrieve(url, destination)
        

    return destination, dims, skip_first

def load_pretrained_embeddings(pretrained_enc_type):
    """
    Load pre-trained embeddings. 
    """
    cache_pkl = f"{pretrained_enc_type}_pretrained_emb.pkl"

    if pretrained_enc_type == 'fasttext':
        if os.path.exists(cache_pkl):
            with open(cache_pkl, "rb") as f:
                pretrained_emb_model = pickle.load(f)
        else:
            emb_dest, embtable_dims, embtable_metadata = download_pretrained(pretrained_enc_type)
            
            pretrained_emb_model = clslib.PretrainedEmbeddings(
                emb_dest, embtable_dims, skip_first_line=embtable_metadata
            )
            with open(cache_pkl, "wb") as f:
                pickle.dump(pretrained_emb_model, f)
        print(f"Loaded pretrained embeddings: {pretrained_enc_type}\n")
    else:
        raise ValueError(f"Unknown pretrained_enc_type: {pretrained_enc_type}")
    return pretrained_emb_model