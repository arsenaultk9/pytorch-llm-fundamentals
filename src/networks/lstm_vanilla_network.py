import torch
import torch.nn as nn

import src.constants as constants

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class LstmVanillaNetwork(nn.Module):

    def __init__(self, tokenizer):
        super(LstmVanillaNetwork, self).__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=tokenizer.vocab_size,
            embedding_dim=constants.EMBED_DIMENSION,
            max_norm=constants.EMBED_MAX_NORM,
            padding_idx=constants.PADDING_IDX
        )

        self.lstm = nn.LSTM(constants.EMBED_DIMENSION, constants.LSTM_HIDDEN_SIZE, constants.LSTM_STACK_COUNT,
                            bidirectional=False, batch_first=True)

        self.linear = nn.Linear(
            in_features=constants.LSTM_HIDDEN_SIZE,
            out_features=tokenizer.vocab_size,
        )

    def get_initial_hidden_context(self):
        h = torch.zeros((constants.LSTM_STACK_COUNT, constants.BATCH_SIZE, constants.LSTM_HIDDEN_SIZE)).to(
            device)  # 1 is for num_layers * 1 for unidirectional lstm
        c = torch.zeros(
            (constants.LSTM_STACK_COUNT, constants.BATCH_SIZE, constants.LSTM_HIDDEN_SIZE)).to(device)

        return (h, c)

    def forward(self, inputs, _, h_c_tupple):
        x = self.embeddings(inputs)
        (x, (h, c)) = self.lstm(x, h_c_tupple)
        x = self.linear(x)

        return x, (h, c)
