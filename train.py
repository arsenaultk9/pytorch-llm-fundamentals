import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import src.constants as constants
from src.lstm_data_loader import get_data
from src.network_trainer import NetworkTrainer
from src.networks.lstm_vanilla_network import LstmVanillaNetwork

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset, valid_dataset, test_dataset = get_data()

train_data_loader = DataLoader(train_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
valid_data_loader = DataLoader(valid_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)
test_data_loader = DataLoader(test_dataset, constants.BATCH_SIZE, constants.SHUFFLE_DATA)

network = LstmVanillaNetwork(tokenizer).to(device)
trainer = NetworkTrainer(network, train_data_loader, valid_data_loader, test_data_loader, constants.EPOCHS, is_dynamic_lr_scheduler=True)

for epoch in range(1, constants.EPOCHS + 1):
    trainer.epoch_train(epoch)
    trainer.epoch_valid(epoch)

trainer.test()

# Turn off training mode & switch to model evaluation
network.eval()