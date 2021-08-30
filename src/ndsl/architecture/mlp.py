import torch
import torch.nn as nn

class FFModel(nn.Module):

    def __init__(self, n_inputs, hidden_sizes, n_outputs, dropouts):
        super(FFModel, self).__init__()
                
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        last_size = n_inputs
        
        for hidden_size, dropout in zip(hidden_sizes, dropouts):
            self.linears.append(nn.Linear(last_size, hidden_size))
            
            self.activations.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            last_size = hidden_size
        
        self.output = nn.Linear(last_size, n_outputs)
        
    def forward(self, inp):

        out = inp
        for lin, drop, act in zip(self.linears, self.dropouts, self.activations):
            out = lin(out)
            out = drop(out)
            out = act(out)
        
        return self.output(out)