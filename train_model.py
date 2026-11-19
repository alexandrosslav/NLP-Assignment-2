from datasets import load_dataset
from collections import OrderedDict

import utils
import clslib
import experiment


#------------------------ hyperparameters and dataset #------------------------

hidden_size1 = 50   # ignored if regressor_type is 'mmlp-regressor'
hidden_size2 = 50   # ignored if regressor_type is 'mmlp-regressor'
dropout = 0.1       # ignored if regressor_type is 'mmlp-regressor'
lstm_hidden_size = 256
lstm_word_dropout = 0.1
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 64
num_epochs = 10
my_seeds = [0]
num_folds = 5
my_cross_validation_sets = []
pretrained_enc_type = 'fasttext' # 'fasttext' or 'glove'
freeze_pretrained_embeddings = True
regressor_type = 'softreg'    #'mmlpreg'
enc_type = 'embed-and-pool'    #'lstm'
device='cpu' #'gpu'
pretrained_emb_model = utils.load_pretrained_embeddings(pretrained_enc_type)

if regressor_type == 'softreg':
    regressor_kwargs={'hidden_size1': hidden_size1, 'hidden_size2': hidden_size2, 'dropout': dropout}
elif regressor_type == 'mmlpreg':
    regressor_kwargs={'hidden_size1': 128, 'hidden_size2': 128, 'dropout': 0.1}

if enc_type == 'lstm':
    enc_kwargs={'freeze_embeddings': freeze_pretrained_embeddings, 'hidden_size': lstm_hidden_size, 'word_dropout': lstm_word_dropout}
    tokenizer = clslib.BasicTokenizer(
        lowercase=True, 
        remove_punct=False,
        remove_stopwords=False, 
        language='english',
        vocab=pretrained_emb_model.vocab
)
elif enc_type == 'embed-and-pool':
    enc_kwargs={'freeze_embeddings': freeze_pretrained_embeddings, 'average_pool': pooling_type == 'mean'}
    tokenizer = clslib.BasicTokenizer(
        lowercase=True, 
        remove_punct=False, 
        remove_stopwords=False, 
        language='english',
        vocab=pretrained_emb_model.vocab
    )

save_to = f"results/{regressor_type}-{enc_type}-{pretrained_enc_type}"

if enc_type == 'embed-and-pool':
    save_to += f"-{pooling_type}"
if not freeze_pretrained_embeddings:
    save_to += "-unfrozen"

my_classifiers = OrderedDict()
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

my_classifiers[enc_type] = []
for run, seed in enumerate(my_seeds): 
    my_classifiers[enc_type].append(
        experiment.cross_validation(
            data_splits=my_cross_validation_sets[run],
            tokenizer=tokenizer,
            emb_model=pretrained_emb_model,
            enc_type=enc_type, 
            enc_kwargs=enc_kwargs,
            regressor_type=regressor_type,
            regressor_kwargs=regressor_kwargs,
            opt_kwargs={'learning_rate': learning_rate, 'weight_decay': weight_decay, 'batch_size': batch_size, 'num_epochs': num_epochs},    
            device=device,
            save_to=f"{save_to}/seed-{seed}"  
        )
    )
