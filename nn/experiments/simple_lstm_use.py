import sys
import os
import pandas as pd
import numpy as np
from torch import nn
import torch 

######### CONSTANTS #########

MODULE_PATH = os.path.dirname(os.path.realpath(__file__)).replace("experiments","")
ROOT_PATH = MODULE_PATH.replace("nn/","")

sys.path.append(MODULE_PATH)

from RNN.LSTM.lstm import LSTM
from train.train import DefaultTrainingProcess

######### CODE #########

def load_dataframe(path_data:str,type_f:str):
    loads = {
        "csv": pd.read_csv,
        "excel": pd.read_excel,
        "parquet": pd.read_parquet
    }

    fc_load = loads[type_f]
    return fc_load(f"{ROOT_PATH}{path_data}")

def build_input(df_dataset:pd.DataFrame):
    df_dataset["past_values"] = [ 
            w.to_list() 
            for w in df_dataset["IPG2211A2N"].rolling(
            window=6, closed='left')
        ]
    df_dataset['valid'] = df_dataset.past_values.apply(lambda l: len(l) == 6)
    df_dataset = df_dataset[df_dataset.valid]
    X = df_dataset.past_values.values
    X = np.array([np.array(x) for x in X])
    y = df_dataset["IPG2211A2N"].values
    X,y = torch.from_numpy(X).float(),torch.from_numpy(y).float()
    X = X.unsqueeze(-1)
    return X,y


class CustomModel(nn.Module):
    def __init__(self, input_shape:tuple ) -> None:
        super(CustomModel,self).__init__()
        self.lstm1 = LSTM(input_shape,32,16)
        self.lstm2 = LSTM((input_shape[0],6,32*16),32,16)
        self.lstm3 = LSTM((input_shape[0],6,32*16),16,8)
        self.linear = nn.Linear(16*8,1)
    
    def forward(self,x):
        sequences, (h,c) = self.lstm1(x)
        sequences, (h,c) = self.lstm2(sequences,(h,c))
        sequences, (h,c) = self.lstm3(sequences)
        output = self.linear(sequences[-1,:,:])
        return output
    
def main():
    df_dataset = load_dataframe("datasets/time_series/Electric_Production.csv","csv")
    df_dataset["DATE"] = pd.to_datetime(df_dataset.DATE)
    df_dataset = df_dataset.sort_values("DATE")
    X,y = build_input(df_dataset)
    X_train,y_train = X[:-36],y[:-36]
    X_val,y_val = X[-36:],y[-36:]


    model = CustomModel(X_train.shape[1:])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9)
    train_pipe = DefaultTrainingProcess(model,criterion,optimizer)
    train_pipe.fit(X_train,y_train,100)

    pass 