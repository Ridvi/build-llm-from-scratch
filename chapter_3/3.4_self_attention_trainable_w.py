import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2=inputs[1]
d_in=inputs.shape[-1]
d_out=2

torch.manual_seed(123)
w_query=torch.nn.Parameter(torch.rand(d_in,d_out))
w_key=torch.nn.Parameter(torch.rand(d_in,d_out))
w_value=torch.nn.Parameter(torch.rand(d_in,d_out))

query_2=x_2@w_query
key_2=x_2@w_key
value_2=x_2@w_value

#for all

keys=inputs@w_key
values=inputs@w_value

attn_scores_2=query_2@keys.T

d_k=keys.shape[-1]
attn_weights_2=torch.softmax(attn_scores_2/d_k**0.5,dim=-1)

context_vec_2=attn_weights_2@values

#listing-3.1 a compact self-attention class
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.w_query=nn.Parameter(torch.rand(d_in,d_out))
        self.w_key=nn.Parameter(torch.rand(d_in,d_out))
        self.w_value=nn.Parameter(torch.rand(d_in,d_out))

    def forward(self,x):
        keys=x@self.w_key
        queries=x@self.w_query
        values=x@self.w_value
        attn_scores=queries@keys.T
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        context_vec=attn_weights@values
        return context_vec
    
torch.manual_seed(123)
sa_v1=SelfAttentionV1(d_in,d_out)

#listing 3.2 self-attention class using pytorch's linear layers

class SelfAttentionV2(nn.Module):
    def __init__(self,d_in,d_out,qkv_bias=False):
        super().__init__()
        self.w_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value=nn.Linear(d_in,d_out,bias=qkv_bias)

    def forward(self,x):
        keys=self.w_key(x)
        queries=self.w_query(x)
        values=self.w_value(x)
        attn_scores=queries@keys.T
        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
        context_vec=attn_weights@values
        return context_vec
    
torch.manual_seed(789)
sa_v2=SelfAttentionV2(d_in,d_out)

# print(sa_v2(inputs))     

#exercise 3.1

sa_v1.w_query = torch.nn.Parameter(sa_v2.w_query.weight.T)
sa_v1.w_key = torch.nn.Parameter(sa_v2.w_key.weight.T)
sa_v1.w_value = torch.nn.Parameter(sa_v2.w_value.weight.T)

print(sa_v1(inputs))
print(sa_v2(inputs))

