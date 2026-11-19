from datasets import load_dataset
from collections import OrderedDict

import utils
import clslib
import experiment
import argparse
import json
from pathlib import Path

# -------------------- command-line arguments --------------------

parser = argparse.ArgumentParser()

# Experiment identification
parser.add_argument("--run-id", type=str, required=True)

# Model configuration
parser.add_argument("--embedding", choices=["fasttext", "glove"],
                    default="fasttext")
parser.add_argument("--encoder", choices=["embed-and-pool", "lstm"],
                    default="embed-and-pool")
parser.add_argument("--pooling", choices=["mean", "sum"], default="mean")
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--word-dropout", type=float, default=0.1)
parser.add_argument("--trainable-embeddings", action="store_true")
parser.add_argument("--regressor", choices=["softreg", "mmlpreg"], default="softreg")

# Training configuration
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")

args = parser.parse_args()

# -------------------- map arguments to existing variables --------------------

hidden_size1 = 128 # ignored if regressor_type is 'mmlp-regressor'
hidden_size2 = 128 # ignored if regressor_type is 'mmlp-regressor'
dropout = 0.1      # ignored if regressor_type is 'mmlp-regressor'

lstm_hidden_size = args.hidden_size # 256, 128, 512
lstm_word_dropout = args.word_dropout # 0.1, 0.2, 0.3

learning_rate = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epochs = args.epochs

my_seeds = [args.seed]
num_folds = 5
my_cross_validation_sets = []

pretrained_enc_type = args.embedding # 'fasttext' or 'glove'
freeze_pretrained_embeddings = not args.trainable_embeddings

regressor_type = args.regressor # 'softreg' or 'mmlpreg'
enc_type = args.encoder # 'embed-and-pool' or 'lstm'
average_pool = args.pooling == "mean" # otherwise uses sum
device = args.device

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
    enc_kwargs={'freeze_embeddings': freeze_pretrained_embeddings, 'average_pool': average_pool}
    tokenizer = clslib.BasicTokenizer(
        lowercase=True, 
        remove_punct=False, 
        remove_stopwords=False,
        language='english',
        vocab=pretrained_emb_model.vocab
    )

# -------------------- experiment output --------------------

if not args.run_id.replace("-", "").replace("_", "").isalnum():
    parser.error("--run-id must contain only letters, numbers, - or _")

if not args.folds or len(set(args.folds)) != len(args.folds):
    parser.error("--folds must contain distinct fold indices")

if any(fold not in range(num_folds) for fold in args.folds):
    parser.error(f"--folds must be between 0 and {num_folds - 1}")

save_to = Path("results") / args.run_id

if save_to.exists():
    parser.error(f"Experiment directory already exists: {save_to}")

save_to.mkdir(parents=True)

config = vars(args).copy()
config["classifier_hidden_sizes"] = [hidden_size1, hidden_size2]
config["classifier_dropout"] = dropout
config["num_folds"] = num_folds

with open(save_to / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"Experiment: {args.run_id}")
print(f"Results directory: {save_to}")

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
