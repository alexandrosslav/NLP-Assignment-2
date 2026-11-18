import torch
from datasets import load_dataset
import os
import pickle
from collections import OrderedDict

import clslib
import utils
import experiment

#------------------------ pre-trained embeddings #------------------------

pretrained_enc_type = 'fasttext'


cache_pkl = f"{pretrained_enc_type}_pretrained_emb.pkl"

if os.path.exists(cache_pkl):
    with open(cache_pkl, "rb") as f:
        pretrained_emb_model = pickle.load(f)
else:
    emb_dest, embtable_dims, embtable_metadata = utils.download_pretrained(pretrained_enc_type)
    
    pretrained_emb_model = clslib.PretrainedEmbeddings(
        emb_dest, embtable_dims, skip_first_line=embtable_metadata
    )
    with open(cache_pkl, "wb") as f:
        pickle.dump(pretrained_emb_model, f)
print(f"Loaded pretrained embeddings: {pretrained_enc_type}\n")

#------------------------ hyperparameters and dataset #------------------------

hidden_size1 = 50
hidden_size2 = 50
dropout = 0.1
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 64
num_epochs = 20
my_seeds = [0]
num_folds = 5
my_cross_validation_sets = []
enc_kwargs={'freeze_embeddings': True}
enc_type = 'embed-and-pool'    #'lstm'
device='cpu' #'gpu'
tokenizer_for_bow = clslib.BasicTokenizer(
    lowercase=True, 
    remove_punct=True, 
    remove_stopwords=True, 
    language='english',
    vocab=pretrained_emb_model.vocab
)
my_classifiers = OrderedDict()
save_to="results/softreg-" + enc_type + "-" + pretrained_enc_type

dataset = load_dataset("stanfordnlp/sst2")
train_dataset = dataset["train"]
test_dataset = dataset["validation"]

print(f"Loaded training dataset with size: {len(train_dataset)}")
print(f"Loaded test dataset with size: {len(test_dataset)}\n")

#------------------------ cross val and training #------------------------

for seed in my_seeds:  
    my_cross_validation_sets.append(
        experiment.cross_validation_split(
            # we pair the texts with their labels
            pairs=list(zip(dataset["train"]["sentence"], dataset["train"]["label"])),
            k=num_folds,
            seed=seed
        )
    )

my_classifiers['embed-and-pool'] = []
for run, seed in enumerate(my_seeds): 
    my_classifiers['embed-and-pool'].append(
        experiment.cross_validation(
            data_splits=my_cross_validation_sets[run],
            tokenizer=tokenizer_for_bow,
            emb_model=pretrained_emb_model,
            enc_type=enc_type, 
            enc_kwargs=enc_kwargs,
            regressor_kwargs={'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2, 'dropout': dropout},
            opt_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay, 'batch_size': batch_size, 'num_epochs': num_epochs},    
            device=device,
            save_to=f"{save_to}/seed-{seed}"  
        )
    )
