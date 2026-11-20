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
            # url = "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip" the link is broken as of now
            url = "https://surfdrive.surf.nl/s/qYfZ6PiNrMLCwpS/download"
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


    if pretrained_enc_type == "glove300":
        import numpy as np
        from huggingface_hub import hf_hub_download

        repo = "jsanzolac/glove_2024_wikigiga_300d"

        vocab_path = hf_hub_download(
            repo_id=repo, filename="glove_vocab.txt"
        )
        vectors_path = hf_hub_download(
            repo_id=repo, filename="glove_vectors.bin"
        )

        with open(vocab_path, encoding="utf-8") as f:
            source_words = [line.rstrip("\r\n").lower() for line in f]

        source_vectors = np.memmap(
            vectors_path,
            dtype=np.float32,
            mode="r",
            shape=(len(source_words), 300)
        )

        # Match the original loader: lowercase and skip duplicates.
        seen = {"<pad>", "<unk>"}
        words, indices = [], []
        for i, word in enumerate(source_words):
            if word not in seen:
                seen.add(word)
                words.append(word)
                indices.append(i)

        vectors = np.empty((len(words) + 2, 300), dtype=np.float32)
        vectors[0] = 0.0  # Padding vector
        if len(indices) == len(source_words):
            vectors[2:] = source_vectors
        else:
            vectors[2:] = source_vectors[indices]
        vectors[1] = vectors[2:].mean(axis=0)  # Unknown vector

        model = clslib.PretrainedEmbeddings.__new__(
            clslib.PretrainedEmbeddings
        )
        model.vocab = clslib.Vocabulary(
            ["<pad>", "<unk>"] + words,
            default="<unk>",
            frozen=True
        )
        model.vectors = vectors
        model.num_dims = 300
        model.pad_token = "<pad>"
        model.unk_token = "<unk>"

        print(f"Loaded GloVe 300D: {len(model.vocab):,} tokens")
        return model


    if pretrained_enc_type in ('fasttext', 'glove'):
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