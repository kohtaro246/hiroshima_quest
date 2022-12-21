import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

conf = {
    "seed": 42,
    "batch_size": 512,
    "num_epoch": 30,
    "dim_model":512,
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

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :]) 

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        in_num_features,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.encoder_input_layer = nn.Linear(in_num_features, dim_model)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        self.decoder_input_layer = nn.Linear(
            in_features=1, # the number of features you want to predict. Usually just 1 
            out_features=dim_model
        ) 
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        self.linear_mapping = nn.Linear(
            in_features=dim_model,
            out_features=1
        )
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None, src_mask=None):
        src = self.encoder_input_layer(src)
        src = self.positional_encoder(src)
        src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )
        decoder_output = self.linear_mapping(decoder_output)
        return decoder_output
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

class HiroshimaQuestDataset(Dataset):
    def __init__(self, sequence):
        self.src = torch.FloatTensor(sequence[:, :conf["src_seq_len"]]).to("cuda")
        self.tgt = torch.FloatTensor(sequence[:, conf["src_seq_len"]-1:-1]).to("cuda")
        self.tgt_y = torch.FloatTensor(sequence[:, conf["src_seq_len"]:]).to("cuda")
    
    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], self.tgt_y[index]

def preprocess_for_training(basepath, start_date=0, end_date=2190, save_sequence=True):
    if save_sequence:
        waterlevel = pd.read_csv(basepath + "waterlevel/data.csv")
        waterlevel_stations = pd.read_csv(basepath + "waterlevel/stations.csv")
        waterlevel = waterlevel[(waterlevel["date"]>=start_date) & (waterlevel["date"]<=end_date)]
        dates = waterlevel["date"].unique().tolist()
        train_days, val_days = train_test_split(dates, test_size=0.2, random_state=conf["seed"])
        waterlevel = waterlevel.sort_values(by=["station", "date"])
        waterlevel_shift = waterlevel.shift(-1).iloc[:-1]
        waterlevel_shift = waterlevel_shift.drop(columns=["date", "station", "river"])
        waterlevel = pd.concat([waterlevel, waterlevel_shift], axis=1)
        waterlevel = waterlevel[waterlevel["date"]!=end_date]
        train_waterlevel = waterlevel[waterlevel["date"].isin(train_days)]
        val_waterlevel = waterlevel[waterlevel["date"].isin(val_days)]
        train_waterlevel = train_waterlevel.drop(columns=["date", "station", "river"])
        train_waterlevel = train_waterlevel.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        train_waterlevel = train_waterlevel.fillna(0.0)
        train_waterlevel = train_waterlevel.astype(float)
        val_waterlevel = val_waterlevel.drop(columns=["date", "station", "river"])
        val_waterlevel = val_waterlevel.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        val_waterlevel = val_waterlevel.fillna(0.0)
        val_waterlevel = val_waterlevel.astype(float)
        train_sequence = train_waterlevel.to_numpy()[:,:,np.newaxis]
        val_sequence = val_waterlevel.to_numpy()[:,:,np.newaxis]
        np.save("train", train_sequence)
        np.save("val", val_sequence)
    else:
        train_sequence = np.load("train.npy")
        val_sequence = np.load("val.npy")
        
    train_dataset = HiroshimaQuestDataset(train_sequence)
    val_dataset = HiroshimaQuestDataset(val_sequence)

    return train_dataset, val_dataset

def create_input_dataframe(input, days):
    # 入力する要素を追加する場合はここを変更！
    df = None
    for day in days:
        data = input[day]
        stations = data['stations']
        waterlevel = data['waterlevel']
        merged = pd.merge(pd.DataFrame(stations, columns=['station']), pd.DataFrame(waterlevel))
        df = pd.concat([df, merged])
    df['value'] = df['value'].replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
    df['value'] = df['value'].fillna(0.0)
    df['value'] = df['value'].astype(float)
    df = df.rename(columns={'value': 'waterlevel'})
    return df


def create_input_sequence(input):
    """Create input data

        Args:
            input: Data of the sample you want to make inference from (dict)

        Returns:
            list: 

        """
    train_val_days = list(input.keys())[:-1]
    train_days, val_days = train_test_split(train_val_days, test_size=0.2, random_state=conf["seed"]) # 本当にこの分け方がいいの？
    sequence = np.zeros((0, 48, len(conf["using_columns"])))
    for day in train_days:
        df = create_input_dataframe(input, [day, day+1])
        for station in df["station"].unique().tolist():
            src_tgt_seq = df.loc[df["station"]==station, conf["using_columns"]].to_numpy()
            sequence = np.concatenate([sequence, src_tgt_seq[None]])
    train_dataset = HiroshimaQuestDataset(sequence)

    sequence = np.zeros((0, 48, len(conf["using_columns"])))
    for day in val_days:
        df = create_input_dataframe(input, [day, day+1])
        for station in df["station"].unique().tolist():
            src_tgt_seq = df.loc[df["station"]==station, conf["using_columns"]].to_numpy()
            sequence = np.concatenate([sequence, src_tgt_seq[None]])
    val_dataset = HiroshimaQuestDataset(sequence)

    return train_dataset, val_dataset

def train_loop(model, opt, loss_fn, train_dataloader, device):
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch, tgt_y_batch in tqdm(train_dataloader):
        src_batch.to(device)
        tgt_batch.to(device)
        tgt_y_batch.to(device)

        sequence_length = tgt_batch.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        pred = model(src_batch, tgt_batch, tgt_mask) 
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
        for src_batch, tgt_batch, tgt_y_batch in tqdm(val_dataloader):
            src_batch.to(device)
            tgt_batch.to(device)
            tgt_y_batch.to(device)

            sequence_length = tgt_batch.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(src_batch, tgt_batch, tgt_mask) 
            loss = loss_fn(pred, tgt_y_batch)
            total_loss += loss.detach().item()
        
    return total_loss / len(val_dataloader)

def train(basepath, start_date=0, end_date=2190, save_sequence=True):
    torch_fix_seed()
    #train_dataset, val_dataset = create_input_sequence(input)
    train_dataset, val_dataset = preprocess_for_training(basepath, start_date, end_date, save_sequence)
    train_dataloader = DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(
        in_num_features=len(conf["using_columns"]), dim_model=conf["dim_model"], num_heads=2, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_loss_list = []
    validation_loss_list = []

    for epoch in range(conf["num_epoch"]):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        print("Training")
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, device)
        train_loss_list += [train_loss]
        print("Validating")
        validation_loss = valid_loop(model, loss_fn, val_dataloader, device)
        validation_loss_list += [validation_loss]
        print(f"Training loss: {math.sqrt(train_loss):.4f}")
        print(f"Validation loss: {math.sqrt(validation_loss):.4f}")

if __name__ == "__main__":
    basepath = "/home/mil/k-tanaka/semi/hiroshima_quest/train/"
    start_date = 0
    end_date = 2190
    save_sequence = True
    train(basepath, start_date, end_date, save_sequence)