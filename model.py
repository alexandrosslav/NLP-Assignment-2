from datasets import load_dataset
from collections import OrderedDict

import utils
import clslib
import experiment


#------------------------ hyperparameters and dataset #------------------------

pretrained_enc_type = 'fasttext'
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
pretrained_emb_model = utils.load_pretrained_embeddings(pretrained_enc_type)
tokenizer = clslib.BasicTokenizer(      # this is for BoW
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

print(f"Loaded training dataset with size: {len(train_dataset)}")

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
            tokenizer=tokenizer,
            emb_model=pretrained_emb_model,
            enc_type=enc_type, 
            enc_kwargs=enc_kwargs,
            regressor_kwargs={'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2, 'dropout': dropout},
            opt_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay, 'batch_size': batch_size, 'num_epochs': num_epochs},    
            device=device,
            save_to=f"{save_to}/seed-{seed}"  
        )
    )
