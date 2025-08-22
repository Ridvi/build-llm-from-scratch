import torch
import torch.nn as nn

GPT_CONFIG_124M={
        'vocab_size':50257,
        "context_length":1024,
        'emb_dim':768,
        'n_heads':12,
        'n_layers':12,
        'drop_rate':0.1,
        "qkv_bias":False
}

#listing gelu activation

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5*x*(1+torch.tanh(
            torch.sqrt(torch.tensor(2.0/torch.pi))*(
                x+0.044715*torch.pow(x,3)
            )
        ))