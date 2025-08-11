import torch.nn as nn
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

queries=sa_v2.w_query(inputs)
keys=sa_v2.w_key(inputs)
attn_scores=queries@keys.T
attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)

context_length=attn_weights.shape[0]
mask_simple=torch.tril(torch.ones(context_length,context_length))

masked_simple=attn_weights*mask_simple

row_sum=masked_simple.sum(dim=-1,keepdim=True)
masked_simple_norm=masked_simple/row_sum

# more efficient way

mask=torch.triu(torch.ones(context_length,context_length),diagonal=1)
masked=attn_scores.masked_fill(mask.bool(),-torch.inf)

attn_weights=torch.softmax(masked/keys.shape[-1]**0.5,dim=-1)

dropout=torch.nn.Dropout(0.5)
#print(dropout(attn_weights))

#3.5.3 implementing a compact causal attention class
batch=torch.stack((inputs,inputs),dim=0)
#print(batch.shape)


#listing 3.3 a compact causal attention class

class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.w_query=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self,x):
        b,num_tokens,d_in=x.shape
        keys=self.w_key(x)
        queries=self.w_query(x)
        values=self.w_value(x)

        atten_scores=queries@keys.transpose(1,2)
        atten_scores.masked_fill_(
            self.mask.bool()[:num_tokens,:num_tokens],-torch.inf
        )
        attn_weights=torch.softmax(atten_scores/keys.shape[-1]**0.5,dim=-1)
        attn_weights=self.dropout(attn_weights)
        context_vector=attn_weights@values
        return context_vector

torch.manual_seed(123)
context_length=batch.shape[1]
ca=CausalAttention(d_in,d_out,context_length,0.0)
context_vecs=ca(batch)
print(context_vecs)



