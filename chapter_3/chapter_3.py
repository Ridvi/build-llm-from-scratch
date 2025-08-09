import torch 

#3.3.1 simple self-attention mechanism without trainable weights
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query=inputs[1]
attn_scores_2=torch.empty(inputs.shape[0])

for i,x_i in enumerate(inputs):
    attn_scores_2[i]=torch.dot(query,x_i)

#normalizing

attn_weights_2=torch.softmax(attn_scores_2,dim=0)

#context vector

query=inputs[1]
ext_vec_2=torch.zeros(query.shapecont)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

#3.3.2 computing attention weights for all input tokens

attn_scores=inputs@inputs.T

#normaliztion
attn_weights=torch.softmax(attn_scores,dim=-1)

#context_vector

all_context_vector=attn_weights@inputs

