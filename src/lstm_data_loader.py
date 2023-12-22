import pickle
import torch

import src.constants as constants
from src.models.lstm_dataset import LstmSequenceDataset
    
def get_inputs_tensor(data):
    inputs = data['input_ids']
    return inputs

def get_data_set(data):
    inputs = data['input_ids']
    attention_masks = data['attention_mask']
    
    Xs = []
    Ys = []
    
    for index, input in enumerate(inputs):
        attention_mask = attention_masks[index]
        mask_stops = [i for i, j in enumerate(attention_mask) if j == 0]
        
        # Skip short sequences
        if len(mask_stops) > 0 and mask_stops[0] < (constants.SEQUENCE_LENGTH - 10):
            continue
        
        x = input[0:-1]
        y = input[1:]
        
        for cur_pos in range(0, len(x) - constants.WINDOW_SLIDE_LENGTH + 1, constants.WINDOW_SLIDE_RANGE):
            end_pos = cur_pos + constants.WINDOW_SLIDE_LENGTH
            
            x_window_slide = x[cur_pos:end_pos]
            y_window_slide = y[cur_pos:end_pos]
        
            Xs.append(x_window_slide)
            Ys.append(y_window_slide)
        
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