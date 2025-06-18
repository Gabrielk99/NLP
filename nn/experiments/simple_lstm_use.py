import sys
import os
import pandas as pd
import numpy as np
from torch import nn
import torch 
import plotly.graph_objects as go
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

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

def build_input(df_dataset:pd.DataFrame,window_s:int,col_t:str):
    df_dataset.set_index("Date",inplace=True)
    scaler = StandardScaler()
    df_dataset[f"{col_t}_norm"] = scaler.fit_transform(df_dataset[col_t].values.reshape(-1,1)).reshape(-1)
    df_dataset["past_values"] = [ 
            w.to_list() 
            for w in df_dataset[col_t].rolling(
            window=window_s, closed='left')
        ]
    df_dataset['valid'] = df_dataset.past_values.apply(lambda l: len(l) == window_s)
    df_dataset = df_dataset[df_dataset.valid]
    X = df_dataset.past_values.values
    X = np.array([np.array(x) for x in X])
    y = df_dataset[col_t].values    
    X,y = torch.from_numpy(X).float(),torch.from_numpy(y).float()
    X = X.unsqueeze(-1)
    
    return df_dataset,X,y,scaler


class CustomModel(nn.Module):
    def __init__(self, input_shape:tuple,torch_lstm:bool = False) -> None:
        super(CustomModel,self).__init__()
        lstm_out_size = None
        if torch_lstm:
            self.lstm1 = nn.LSTM(input_shape[-1],64)
            lstm_out_size = 64
        else:
            units = 1
            self.lstm1 = LSTM(input_shape,units,64)
            lstm_out_size = 64*units
        self.linear1 = nn.Linear(lstm_out_size,128)
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,32)
        self.out = nn.Linear(32,1)

        self.relu = nn.ReLU()
        
    def forward(self,x):
        sequences, (h,c) = self.lstm1(x)
        output = self.linear1(sequences[:,-1,:])
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.out(output)
        output = output.reshape(output.shape[0])
        return output
    
    def predict(self,x):
        outputs = []

        x_0 = x[0]
        x = x[1:]
        output = self.forward(x_0.unsqueeze(0))
        outputs.extend(output)

        max_feat_size = x.shape[1]
        for x_i in x:
            step_left = min(len(outputs),max_feat_size)
            x_i[-step_left:] = torch.stack(outputs[-step_left:]).unsqueeze(1)
            output = self.forward(x_i.unsqueeze(0))
            outputs.extend(output)

        return outputs
            

    
def main():
    df_dataset = load_dataframe("datasets/time_series/daily-minimum-temperatures-in-me.csv","csv")
    df_dataset["Date"] = pd.to_datetime(df_dataset.Date)
    df_dataset = df_dataset.sort_values("Date")
    df_dataset['Daily minimum temperatures'] = df_dataset["Daily minimum temperatures"].str.replace(r'[^\d\.]','',regex=True).astype('float')
    df_dataset,X,y,scaler = build_input(df_dataset,30,"Daily minimum temperatures")
    X_train,y_train = X[:-36],y[:-36]
    X_val,y_val = X[-36:],y[-36:]



    model = CustomModel(X_train.shape[1:],False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_pipe = DefaultTrainingProcess(model,criterion,optimizer)
    train_pipe.fit(X_train,y_train,100)

    train_pipe.model.eval()
    with torch.no_grad():   
        y_v_pred = train_pipe.model.predict(X_val)
    index_val = df_dataset.iloc[-36:].index
    df_dataset.loc[index_val,"y_pred"] = y_v_pred
    r2 = r2_score(y_val,y_v_pred)
    mae = mean_absolute_error(y_val,y_v_pred)
    mape = mean_absolute_percentage_error(y_val,y_v_pred,sample_weight=y_val)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_dataset.index, 
        y=df_dataset["Daily minimum temperatures"], 
        mode='lines+markers',
        name='Real'
    ))
    fig.add_trace(go.Scatter(
        x=df_dataset.index, 
        y=df_dataset["y_pred"], 
        mode='lines+markers',
        name='Previsão do modelo'
    ))
    fig.update_layout(
        title = f"Forecast simple LSTM arch - r2: {r2}; mae: {mae}; mape: {mape}"
    )
    fig.show()

    import ipdb;ipdb.set_trace()