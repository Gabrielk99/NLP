import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm 
    
class DefaultTrainingProcess:
    def __init__(self,model:nn.Module,criterion:nn.Module,
                 optimizer:optim.Optimizer):
        """
        Default class to train any model

        Args:
            model (nn.Module): model to apply the training 
            criterion (nn.Module): the loss criterion to backprop and optimizer
            optimizer (optim.Optimizer): the optimizer algorithm to update weights
        """        
        super(DefaultTrainingProcess,self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self,X:torch.Tensor,y:torch.Tensor,epochs:int,
            batch_size:int=32,shuffle_dataset:bool=False):
        
        dataset = TensorDataset(X,y)
        dataloader = DataLoader(
                        dataset,batch_size=batch_size,
                        shuffle=shuffle_dataset
                    )
        
        loop = tqdm(range(epochs), total=epochs, leave=True)
        for epoch in loop:
            for batch_x,batch_y in dataloader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs,batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(loss=loss.item())
