import torch
from torch import nn
import copy

class Bidirectional (nn.Module):
    def __init__(self, RNNLayer:nn.Module):
        super(self,Bidirectional).__init__()
        self.RNNLayerL = RNNLayer
        self.RNNLayerR = copy.deepcopy(RNNLayer)

    def forward(self,X:torch.tensor,H_Cs:tuple=None):
        H_C_l, H_C_r = H_Cs if H_Cs is not None else (None,None),(None,None)
        outputs_l, H_C_l = self.RNNLayerL.forward(X,H_C_l)
        outputs_r, H_C_r = self.RNNLayerR.forward(X.flip(1),H_C_r)
        
        return torch.concat([outputs_l,outputs_r.flip(1)],axis=-1),H_C_l,H_C_r
    
    