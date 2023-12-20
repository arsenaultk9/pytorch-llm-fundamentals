import pickle
import torch

from src.models.lstm_dataset import LstmSequenceDataset
    
def get_inputs_tensor(data):
    inputs = data['input_ids']
    return inputs

def get_data_set(data):
    inputs = get_inputs_tensor(data)
    
    Xs = []
    Ys = []
    
    for input in inputs:
        x = input[0:-1]
        y = input[1:]
        
        Xs.append(x)
        Ys.append(y)
        
    Xs = torch.Tensor(Xs)
    Ys = torch.Tensor(Ys)
    
    return LstmSequenceDataset(Xs, Ys)

def get_data():
    with open('./data/scitechnews.pkl', 'rb') as file:
        dataset =  pickle.load(file, encoding="latin1")
        
    train_dataset = get_data_set(dataset['train'])
    valid_dataset = get_data_set(dataset['validation'])
    test_dataset = get_data_set(dataset['test'])
    
    return (train_dataset, valid_dataset, test_dataset)