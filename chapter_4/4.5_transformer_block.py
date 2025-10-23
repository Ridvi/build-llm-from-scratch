from multi_head_attention import MultiHeadAttention
from gelu_activation import FeedForward
from layer_normalization import LayerNorm
import torch
import torch.nn as nn

GPT_CONFIG_124M={
        'vocab_size':50257,
        "context_length":1024,
        'emb_dim':768,
        'n_layers':12,
        'n_heads':12,
        'drop_rate':0.1,
        "qkv_bias":False
}

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn=MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg['emb_dim'])
        self.drop_shortcut=nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.drop_shortcut(x)
        x=x+shortcut
        return x
    

torch.manual_seed(123)
x=torch.rand(2,4,768)
block=TransformerBlock(GPT_CONFIG_124M)
output=block(x)

#print(x.shape)


#4.6 coding the gpt model


#4.7 the gpt model architecture implementation
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb=nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.drop_emb=nn.Dropout(cfg['drop_rate'])


        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm=LayerNorm(cfg['emb_dim'])
        self.out_head=nn.Linear(
            cfg['emb_dim'],cfg["vocab_size"],bias=False
        )


    def forward(self,in_idx):
        batch_size,seq_ln=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_embeds=self.pos_emb(
            torch.arange(seq_ln,device=in_idx.device)

        )

        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits
    

torch.manual_seed(123)
model=GPTModel(GPT_CONFIG_124M)

import tiktoken

tokenizer=tiktoken.get_encoding('gpt2')

batch=[]
txt1='Every effort moves you'
txt2='Every day holds a'

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch=torch.stack(batch,dim=0)

out=model(batch)

print(out.shape)