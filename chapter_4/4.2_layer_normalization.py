import tiktoken
import torch 
import torch.nn as nn

tokenizer=tiktoken.get_encoding('gpt2')

batch=[]
txt1='Every effort moves you'
txt2='Every day holds a'

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch=torch.stack(batch,dim=0)

torch.manual_seed(123)
batch_example=torch.randn(2,5)
layer=nn.Sequential(nn.Linear(5,6),nn.ReLU())
out=layer(batch_example)
print(out)

#4.2 #layer normalization
