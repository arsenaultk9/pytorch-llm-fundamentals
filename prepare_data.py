import pickle

import src.constants as constants
from datasets import load_dataset
from transformers import GPT2Tokenizer

dataset = load_dataset("ronaldahmed/scitechnews")
print(dataset)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encode(documents):
    return tokenizer(documents['pr-article'])

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