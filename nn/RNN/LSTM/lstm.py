
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self,input_shape:tuple,units:int,hidden_layers:int,sigma=0.01):
        """
        Custom LSTM implementation

        Args:
            input_shape (tuple): shape from input for the layer
                                the format have to be (?,timesteps,features)
            units (int): how much LSTM units stack                                
            hidden_layers (int): how much "neurons" inside LSTM layer 
            sigma (float, optional): std force parameter, Defaults to 0.01.
        """         
        super(LSTM,self).__init__()
        
        self.input_shape = input_shape
        self.units = units
        self.hidden_layers = hidden_layers
        self.sigma = sigma

        weights_initialization = lambda shape: nn.Parameter(
                                    torch.randn(*shape)*sigma
                                )
        weights = lambda: (
                    (
                        weights_initialization((units,input_shape[-1]+hidden_layers,
                                                hidden_layers)),
                        nn.Parameter(torch.zeros(units,1,hidden_layers))
                    )
                )
        
        self.W_i,self.b_i = weights()
        self.W_f,self.b_f = weights()
        self.W_o,self.b_o = weights()
        self.W_c,self.b_c = weights()

    def forward(self,X,H_C=None):
        if H_C is None:
            H = torch.zeros((self.units,X.shape[0],self.hidden_layers),
                            device=X.device)
            C = torch.zeros((self.units,X.shape[0],self.hidden_layers),
                            device=X.device)
        else: 
            H,C = H_C
        
        sequences = []
        for time_step in range(X.shape[1]):
            X_t = X[:,time_step,:]
            X_t = X_t.unsqueeze(0).expand(self.units,-1,-1)
            try:
                Z = torch.cat([H,X_t], dim=2).float()
                I = torch.sigmoid(torch.matmul(Z,self.W_i)+self.b_i)
                F = torch.sigmoid(torch.matmul(Z,self.W_f)+self.b_f)
                O = torch.sigmoid(torch.matmul(Z,self.W_o)+self.b_o)
                C_cand = torch.tanh(torch.matmul(Z,self.W_c)+self.b_c)
                C = F*C + I*C_cand
                H = O*torch.tanh(C)
                sequences.append(H.transpose(0,1).reshape(X.shape[0],-1))
            except:
                import ipdb;ipdb.set_trace()
        
        return torch.stack(sequences,dim=1), (H,C)

