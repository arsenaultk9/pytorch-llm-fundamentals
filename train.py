import torch
from torch.utils.data import DataLoader

import src.constants as constants
from src.lstm_data_loader import get_data
    
train_dataset, valid_dataset, test_dataset = get_data()


train_data_loader = DataLoader(train_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
valid_data_loader = DataLoader(valid_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
test_data_loader = DataLoader(test_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

print(train_data_loader)

# TODO: Rendu à créer le modèle et l'entrainer.