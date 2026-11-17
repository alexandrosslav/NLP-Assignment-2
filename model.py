import torch
from datasets import load_dataset

import clslib
import utils
import experiment

#------------------------ pre-trained embeddings #------------------------
pretrained_enc_type = 'fasttext'


emb_dest, embtable_dims, embtable_metadata = utils.download_pretrained(pretrained_enc_type)
emb_dest, embtable_dims, embtable_metadata

pretrained_emb_model = clslib.PretrainedEmbeddings(emb_dest, embtable_dims, skip_first_line=embtable_metadata)

#------------------------ hyperparameters and dataset #------------------------

hidden_size1 = 50,
hidden_size2 = 50,
dropout = 0.1,
learning_rate = 1e-3,
weight_decay = 1e-4,
batch_size = 64,
num_epochs = 20,
my_seeds = [0],
num_folds = 5,
my_cross_validation_sets = [],
enc_kwargs={'freeze_embeddings': True},
enc_type = 'embed-and-pool',    #'lstm'
device='cpu' #'gpu',
tokenizer_for_bow = clslib.BasicTokenizer(
    lowercase=True, 
    remove_punct=True, 
    remove_stopwords=True, 
    language='english',
    vocab=pretrained_emb_model.vocab
)

dataset = load_dataset("stanfordnlp/sst2")
train_dataset = dataset["train"]
test_dataset = dataset["validation"]

#------------------------ cross val and training #------------------------

for seed in my_seeds:  
    my_cross_validation_sets.append(
        experiment.cross_validation_split(
            # we pair the texts with their labels
            pairs=list(zip(dataset["train"]["text"], dataset["train"]["label"])),
            k=num_folds,
            seed=seed
        )
    )

metrics, best_epoch = experiment.train(
    my_cross_validation_sets[0][0]['training'], 
    my_cross_validation_sets[0][0]['validation'],
    tokenizer_for_bow,
    pretrained_emb_model,
    num_classes=1,
    save_to="results/softreg-" + enc_type + "-" + pretrained_enc_type,   
    enc_kwargs=enc_kwargs, 
    enc_type=enc_type,
    regressor_kwargs={'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2, 'dropout': dropout},
    opt_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay, 'batch_size': batch_size, 'num_epochs': num_epochs},    
    device=device
)

#Statistics for the best epoch
best_epoch_metrics = metrics[best_epoch]
print(f"Best epoch: {best_epoch}")
print(f"Train accuracy: {best_epoch_metrics['train_metrics']['accuracy']}")
print(f"Validation accuracy: {best_epoch_metrics['val_metrics']['accuracy']}")
