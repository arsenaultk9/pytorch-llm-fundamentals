import torch
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class LstmSequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long, device=device) 
        self.Y = torch.tensor(Y, dtype=torch.long, device=device)

        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        return (X, Y)
