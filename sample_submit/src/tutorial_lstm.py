import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import math

from tutorial_transformer import create_input_sequence

conf = {
    "seed": 42,
    "batch_size": 16,
    "num_epoch": 30,
    "dim_model":32,
    "using_columns": ["waterlevel"],
    "src_seq_len": 24,
    "tgt_seq_len": 24
}

def torch_fix_seed():
    seed = conf["seed"]
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class LSTMNet(nn.Module):
    def __init__(self, in_num_features, dim_model, output_size, hidden_size, batch_first):
        super(LSTMNet, self).__init__()
        self.encoder_input_layer = nn.Linear(in_num_features, dim_model)
        self.rnn = nn.LSTM(input_size = dim_model,
                            hidden_size = hidden_size,
                            batch_first = batch_first)
        self.output_layer = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, inputs):
        inputs = self.encoder_input_layer(inputs)
        h, _= self.rnn(inputs)      
        output = self.output_layer(h)

        return output

def train_loop(model, opt, loss_fn, train_dataloader, device):
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch, tgt_y_batch in train_dataloader:
        src_batch.to(device)
        tgt_batch.to(device)
        tgt_y_batch.to(device)
        pred = model(tgt_batch)
        loss = loss_fn(pred, tgt_y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.detach().item()
    return epoch_loss / len(train_dataloader)

def valid_loop(model, loss_fn, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch, tgt_y_batch in val_dataloader:
            src_batch.to(device)
            tgt_batch.to(device)
            tgt_y_batch.to(device)


            pred = model(tgt_batch) 
            loss = loss_fn(pred, tgt_y_batch)
            total_loss += loss.detach().item()
        
    return total_loss / len(val_dataloader)

def train(input):
    torch_fix_seed()
    train_dataset, val_dataset = create_input_sequence(input)
    train_dataloader = DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMNet(
        in_num_features=len(conf["using_columns"]), dim_model=conf["dim_model"], output_size=1, hidden_size=64, batch_first=True
    ).to(device)
    print(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_loss_list = []
    validation_loss_list = []

    for epoch in range(conf["num_epoch"]):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]
        validation_loss = valid_loop(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]
        print(f"Training loss: {math.sqrt(train_loss):.4f}")
        print(f"Validation loss: {math.sqrt(validation_loss):.4f}")

if __name__ == "__main__":
    train()