import pickle

from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("ronaldahmed/scitechnews")
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(documents):
    return tokenizer(documents['pr-article'], truncation=True, padding="max_length")

train = encode(dataset['train'][0:100])
validation = encode(dataset['validation'][0:15])
test = encode(dataset['test'][0:20])

data = {
    "train": train,
    "validation": validation,
    "test": test
}

with open('./data/scitechnews.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)