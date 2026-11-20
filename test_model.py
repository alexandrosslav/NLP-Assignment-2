# we will be using textstat as a way to otain some descriptive features of text
from datasets import load_dataset
from collections import OrderedDict
from sklearn.metrics import classification_report
from pathlib import Path
import textstat  
import pandas as pd
import numpy as np

import utils
import clslib
import experiment
import json
import argparse

#------------------------ hyperparameters and dataset #------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=str, required=True)
args = parser.parse_args()

model_family = args.run_id

with open(f"results/{model_family}/config.json", "r") as f:
    config = json.load(f)

batch_size = config["batch_size"]
my_seeds = [config["seed"]]
num_folds = config["num_folds"]

pretrained_enc_type = config["embedding"]
freeze_pretrained_embeddings = not config["trainable_embeddings"]
regressor_type = config["regressor"]
enc_type = config["encoder"]

device = 'cpu'

pretrained_emb_model = utils.load_pretrained_embeddings(pretrained_enc_type)

if enc_type == 'lstm':
    tokenizer = clslib.BasicTokenizer(
        lowercase=True, 
        remove_punct=False,
        remove_stopwords=False, 
        language='english',
        vocab=pretrained_emb_model.vocab
)
elif enc_type == 'embed-and-pool':
    tokenizer = clslib.BasicTokenizer(
        lowercase=True, 
        remove_punct=False, 
        remove_stopwords=False, 
        language='english',
        vocab=pretrained_emb_model.vocab
    )

my_classifiers = OrderedDict()
dataset = load_dataset("stanfordnlp/sst2")
test_dataset = dataset["validation"]

print(f"Loaded test dataset with size: {len(test_dataset)}\n")

#------------------------ testing #------------------------

table = []
systems = dict()
items = dict()
metrics = []

for run, seed in enumerate(my_seeds):
    run_texts = [text for text in test_dataset["sentence"]]
    run_labels = np.array([label for label in test_dataset["label"]], dtype=int)

    for fold in range(num_folds):            
        # load our model
        trained_nn = experiment.load(f"results/{model_family}/seed-{seed}/fold-{fold}/model", pretrained_emb_model, regressor_type=regressor_type)
        # test it on the test set
        output = experiment.predict_logits(
            trained_nn, 
            tokenizer, 
            run_texts, 
            batch_size, 
            device
        )
        predictions = (output >= 0).squeeze(1) # get the predicted class
        print(f"Run {run}, fold {fold}:\n{classification_report(run_labels, predictions)}\n")

        report = classification_report(run_labels, predictions, output_dict=True)

        metrics.append({
            "run": run,
            "fold": fold,
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"]
        })

        # then featurize the texts and compute the correctness flags
        for i in range(len(predictions)):                
                            
            # find an int id for each item
            item_id = i

            # text, pred, and label
            x = run_texts[i]
            y_ = predictions[i]
            y = run_labels[i]

            # put it all in a table, along with some features
            table.append([
                model_family, 
                item_id, 
                x,
                run, 
                fold, 
                y, 
                y_, 
                int(y == y_), # correctness flag
                # some features
                textstat.lexicon_count(x),
                textstat.flesch_reading_ease(x),
                textstat.reading_time(x),
                textstat.flesch_kincaid_grade(x)
            ])

df = pd.DataFrame(table, columns=['system', 'item_id', 'sentence', 'run', 'fold',  'label', 'pred', 'correct', 'lexicon_count', 'flesch_reading_ease', 'reading_time', 'flesch_kincaid_grade'])
print(df)

results_dir = Path("results") / model_family

df.to_csv(results_dir / "heldout_predictions.csv", index=False)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(results_dir / "heldout_metrics.csv", index=False)

print(f"Held-out results saved to {results_dir}")