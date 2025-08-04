import tiktoken
import torch 
tokenizer = tiktoken.get_encoding("gpt2")

with open('chapter_2/the-verdict.txt','r',encoding='utf-8') as f:
    text=f.read()

enc_text = tokenizer.encode(text)

enc_sample= enc_text[50:]

context_size=4

for i in range(1,context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    print(context,'---->',desired)


for i in range(1,context_size+1):
    context=enc_sample[:i]
    desired=enc_sample[i]
    print(tokenizer.decode(context),"---->",tokenizer.decode([desired]))


print(torch.backends.mps.is_available())    