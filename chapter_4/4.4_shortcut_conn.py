import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5*x*(1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*(
                x+0.044715*torch.pow(x,3)
            )
        ))
    


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self,layer_sizes,use_shortcut):
        super().__init__()
        self.use_shortcut=use_shortcut
        self.layers=nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]),
                           GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]),
                           GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]),
                           GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]),
                           GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4],layer_sizes[5]),
                           GELU())            
            ]
        )

def forward(self,x):
    for layer in self.layers:
        layer_output=layer(x)
        if self.use_shortcut and x.shape == layer.output_shape:
            x = x+layer_output
        else:
            x=layer_output
        return x