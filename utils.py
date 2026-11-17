from pathlib import Path
import urllib.request

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