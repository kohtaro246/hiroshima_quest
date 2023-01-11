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
import os
import glob

# パラメータ
conf = {
    "seed": 42,                            # ランダム値の生成を固定する
    "batch_size": 512,                     # １つのバッチに含まれるデータ数
    "num_epoch": 30,                       # エポック数
    "dim_model": 64,                       # モデルの次元。次元が大きいほど複雑なパターンを学習できる一方、大きくしすぎると過学習したり、推論に時間がかかる
    "num_features": 3,                     # 入力するデータの種類の数
    "src_seq_len": 24,                     # 入力するデータの時系列方向の次元。24時間分データであれば24次元
    "tgt_seq_len": 24,                     # 出力するデータの時系列方向の次元。24時間分データであれば24次元
    "dropout": 0.1,                        # 過学習を抑えるため、一部の重みを０にする割合。0.1なら1割の重みを0にする
    "test_size": 0.2,                      # validation setの割合。0.2の場合、全データのうち８割が
    "learning_rate": 0.0005,               # 学習率
    "num_heads": 2,                        # transformerのhead数
    "num_layers": 3                        # transformerのlayer数
}

def torch_fix_seed():
    """
    ランダム値の生成を固定する。
    """
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
    """
    モデルに入力するデータの時系列的順番に関する情報を埋め込むためのクラス。変更する必要はあまりないと思われる。
    """
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    """
    モデルを定義するクラス
    """
    # Constructor
    def __init__(
        self,
        in_num_features,
        dim_model,
        num_heads,
        num_encoder_layers,
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
        self.linear_mapping = nn.Linear(
            in_features=dim_model,
            out_features=1
        )

    def forward(self, src, tgt=None, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None, src_mask=None):
        src = self.encoder_input_layer(src)
        src = self.positional_encoder(src)
        src = self.encoder(src=src)
        output = self.linear_mapping(src)
        return output.squeeze(-1)

class HiroshimaQuestDataset(Dataset):
    """
    dataset classの作成。前処理出力形式を変更しない限りはいじる必要はない
    """
    def __init__(self, src, tgt, device):
        """
        src: (データ数、num_features, src_seq_len)
        tgt: (データ数, tgt_seq_len)
        """
        assert src.shape == (tgt.shape[0], conf["num_features"], conf["src_seq_len"])
        assert tgt.shape[1] == conf["tgt_seq_len"]
        self.src = torch.FloatTensor(src).permute(0,2,1).to(device)
        self.tgt = torch.FloatTensor(tgt).to(device)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

def preprocess_for_training(basepath, device, start_date=0, end_date=2190, save_sequence=True):
    """
    データの前処理


    """
    def _missing_value_process(df):
        """
        欠損値処理・型変換・重複削除
        """
        df = df.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        df = df.fillna(0.0)
        hour_columns = list(df.columns[3:])
        type_conversion_dict = {k: float for k in hour_columns}
        df = df.astype(type_conversion_dict)
        #df = df.drop_duplicates()
        return df

    def _src_tgt_using(df, start_date, end_date):
        """
        "src_using"というカラムがTrueの行は訓練時の入力に用い、"tgt_using"というカラムがTrueの行は教師データに使うように、2つのカラムを追加する。
        使う行と使わない行が発生するのは以下の２つの理由による。
        ・各観測所で抜けている日にちがある場合がある。0~2190のうち1454だけ抜けていた場合、訓練時の入力に1453は使わず、教師データで1455は使わない。
        ・0日目は教師データとしては使わない。一方、2190日目は訓練時の入力には使わない。
        """

        df = df.sort_values(by=["station", "date"])
        date_list = df["date"].to_numpy()[:-1]
        shifted_date_list = df["date"].to_numpy()[1:]
        diff = shifted_date_list - date_list - 1
        src_using_list = np.concatenate([np.where(diff==0, True, False), np.array([False])])
        tgt_using_list = np.concatenate([np.array([False]), np.where(diff==0, True, False)])
        df["src_using"] = src_using_list
        df["tgt_using"] = tgt_using_list
        return df

    def _add_rainfall_data(waterlevel, rainfall: pd.DataFrame):
        """
        全観測地点の雨量の平均をデータに追加
        """
        rainfall = rainfall.drop(columns=["station", "city"])
        rainfall = rainfall.groupby("date").mean()
        columns = {col: 'rainfall_' + col for col in rainfall.columns}
        rainfall = rainfall.rename(columns=columns)
        waterlevel_rainfall = pd.merge(waterlevel, rainfall, on=["date"], how="left")
        waterlevel_rainfall = waterlevel_rainfall.sort_values(by=["station", "date"])
        waterlevel_rainfall = waterlevel_rainfall.fillna(0.0)
        return waterlevel_rainfall

    def _add_tidelevel_data(waterlevel, tidelevel):
        """
        全観測地点の水位の平均をデータに追加
        """
        tidelevel = tidelevel.drop(columns=["station", "city"])
        tidelevel = tidelevel.groupby("date").mean()
        columns = {col: 'tidelevel_' + col for col in tidelevel.columns}
        tidelevel = tidelevel.rename(columns=columns)
        waterlevel_tidelevel = pd.merge(waterlevel, tidelevel, on=["date"], how="left")
        waterlevel_tidelevel = waterlevel_tidelevel.sort_values(by=["station", "date"])
        waterlevel_tidelevel = waterlevel_tidelevel.fillna(0.0)
        return waterlevel_tidelevel

    if save_sequence:
        # データの読み込み・欠損値処理
        print("Processing waterlevel data")
        waterlevel = pd.read_csv(basepath + "waterlevel/data.csv")
        waterlevel_stations = pd.read_csv(basepath + "waterlevel/stations.csv")
        waterlevel = waterlevel[(waterlevel["date"]>=start_date) & (waterlevel["date"]<=end_date)]
        waterlevel = _missing_value_process(waterlevel)
        waterlevel = _src_tgt_using(waterlevel, start_date, end_date)

        print("Processing rainfall data")
        rainfall = pd.read_csv(basepath + "rainfall/data.csv")
        rainfall_stations = pd.read_csv(basepath + "rainfall/stations.csv")
        rainfall = rainfall[(rainfall["date"]>=start_date) & (rainfall["date"]<=end_date)]
        rainfall = _missing_value_process(rainfall)
        rainfall = rainfall.drop_duplicates()

        print("Processing tidelevel data")
        tidelevel = pd.read_csv(basepath + "tidelevel/data.csv")
        tidelevel_stations = pd.read_csv(basepath + "tidelevel/stations.csv")
        tidelevel = tidelevel[(tidelevel["date"]>=start_date) & (tidelevel["date"]<=end_date)]
        tidelevel = _missing_value_process(tidelevel)
        tidelevel = tidelevel.drop_duplicates()

        # 入力データの作成 - 入力データを追加したければここにコードを追記する
        # 例として、1つの降水量観測所('栗谷')の降雨量データと水位データを入力とするコードを作成する
        print("Creating input data")
        input_data = _add_tidelevel_data(waterlevel, tidelevel)
        input_data = _add_rainfall_data(input_data, rainfall)  # 入力項目を追加する
        input_data = input_data.fillna(0.0)                    # rainfallで欠損している日にちは0埋めする
        input_data = input_data[input_data["src_using"]]       # 訓練時の入力で使用するデータのみ抽出
        print(input_data)
        print(input_data.columns)

        # train setとvalidation setに分ける
        dates = input_data["date"].unique().tolist()
        src_train_days, _ = train_test_split(dates, test_size=conf["test_size"], random_state=conf["seed"])      # 日にちでtrain setとvalidation setに分ける。
        input_data["train"] = input_data["date"].isin(src_train_days)
        tgt_train_days = np.array(src_train_days) + 1
        waterlevel["train"] = waterlevel["date"].isin(tgt_train_days)

        input_dataframe = input_data #debug
        train_input_data = input_data[input_data["train"]]
        train_input_dataframe =  train_input_data #debug
        val_input_data = input_data[~input_data["train"]]
        dropped_train_input_data = train_input_data.drop(columns=["date", "station", "river", "src_using", "tgt_using", "train"]).to_numpy()
        dropped_val_input_data = val_input_data.drop(columns=["date", "station", "river", "src_using", "tgt_using", "train"]).to_numpy()
        conf["num_features"] = dropped_train_input_data.shape[1]//24
        train_input_data = np.reshape(dropped_train_input_data, (-1,conf["num_features"],24))
        val_input_data = np.reshape(dropped_val_input_data, (-1,conf["num_features"],24))
        print(train_input_data.shape)

        # 教師データの作成
        print("Creating target data")
        tgt_data = waterlevel[waterlevel["tgt_using"]]
        tgt_dataframe = tgt_data #debug
        train_tgt_data = tgt_data[tgt_data["train"]]
        train_tgt_dataframe = train_tgt_data #debug
        val_tgt_data = tgt_data[~tgt_data["train"]]
        train_tgt_data = train_tgt_data.drop(columns=["date", "station", "river", "src_using", "tgt_using", "train"]).to_numpy()
        val_tgt_data = val_tgt_data.drop(columns=["date", "station", "river", "src_using", "tgt_using", "train"]).to_numpy()

        # dataの保存
        np.save("../../train_input_data", train_input_data)
        np.save("../../val_input_data", val_input_data)
        np.save("../../train_tgt_data", train_tgt_data)
        np.save("../../val_tgt_data", val_tgt_data)

    else:
        print("Loading saved data")
        train_input_data = np.load("../../train_input_data.npy")
        val_input_data = np.load("../../val_input_data.npy")
        train_tgt_data = np.load("../../train_tgt_data.npy")
        val_tgt_data = np.load("../../val_tgt_data.npy")

    # datasetのインスタンス化
    print("Instantiate dataset")
    train_dataset = HiroshimaQuestDataset(train_input_data, train_tgt_data, device)
    val_dataset = HiroshimaQuestDataset(val_input_data, val_tgt_data, device)

    return train_dataset, val_dataset

def train_loop(model, opt, loss_fn, train_dataloader, device):
    model.train()
    epoch_loss = 0
    for src_batch, tgt_batch in tqdm(train_dataloader):
        src_batch.to(device)
        tgt_batch.to(device)

        pred = model(src_batch)
        loss = loss_fn(pred, tgt_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.detach().item()
    return epoch_loss / len(train_dataloader)

def valid_loop(model, loss_fn, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(val_dataloader):
            src_batch.to(device)
            tgt_batch.to(device)

            pred = model(src_batch)
            loss = loss_fn(pred, tgt_batch)
            total_loss += loss.detach().item()

    return total_loss / len(val_dataloader)

def train(basepath, device, start_date=0, end_date=2190, save_sequence=True):
    torch_fix_seed()
    train_dataset, val_dataset = preprocess_for_training(basepath, device, start_date, end_date, save_sequence)
    train_dataloader = DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=conf["batch_size"], shuffle=True)

    model = Transformer(
        in_num_features=conf["num_features"], dim_model=conf["dim_model"], num_heads=conf["num_heads"], num_encoder_layers=conf["num_layers"], dropout_p=conf["dropout"]
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=conf["learning_rate"])
    loss_fn = nn.MSELoss()

    train_loss_list = []
    validation_loss_list = []
    min_validation_loss = 999999999.9
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
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            model_list = glob.glob("../model/*")
            for prev_model in model_list:
                os.remove(prev_model)
            torch.save(model.state_dict(), "../model/best.pt")

def make_prediction(input_data, model_path, device):
    model = Transformer(
        in_num_features=conf["num_features"], dim_model=conf["dim_model"], num_heads=conf["num_heads"], num_encoder_layers=conf["num_layers"], dropout_p=conf["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    input_tensor = torch.FloatTensor(input_data).to(device)
    with torch.no_grad():
        pred = model(input_tensor)
    return torch.flatten(pred).tolist()



if __name__ == "__main__":
    basepath = "/home/mil/k-tanaka/semi/hiroshima_quest/train/"
    start_date = 0
    end_date = 2190
    save_sequence = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(basepath, device, start_date, end_date, save_sequence)