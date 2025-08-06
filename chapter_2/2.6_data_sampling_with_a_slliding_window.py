import tiktoken
import torch 
from torch.utils.data import Dataset,DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

with open('chapter_2/the-verdict.txt','r',encoding='utf-8') as f:
    text=f.read()

# enc_text = tokenizer.encode(text)

# enc_sample= enc_text[50:]

# context_size=4

# for i in range(1,context_size+1):
#     context=enc_sample[:i]
#     desired=enc_sample[i]
#     print(context,'---->',desired)


# for i in range(1,context_size+1):
#     context=enc_sample[:i]
#     desired=enc_sample[i]
#     print(tokenizer.decode(context),"---->",tokenizer.decode([desired]))

#listing 2.5: A dataset for bathed inputs and targets
class GPTDatasetV1(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]

        token_ids=tokenizer.encode(text)

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    


def create_dataloader_v1(text,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):

    tokenizer=tiktoken.get_encoding("gpt2")

    dataset=GPTDatasetV1(text,tokenizer,max_length,stride)

    dataloader=DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader



# dataloader=create_dataloader_v1(text,batch_size=1,max_length=8,stride=8,shuffle=False)

# data_iter=iter(dataloader)
# first_batch=next(data_iter)
# second_batch=next(data_iter)
# print(first_batch)
# print(second_batch)


vocab_size=50257
output_dim=256

token_embedding_layeer=torch.nn.Embedding(vocab_size,output_dim)

max_length=4
dataloader=create_dataloader_v1(text,batch_size=8,max_length=max_length,stride=max_length,shuffle=False)

data_iter=iter(dataloader)

inputs,target=next(data_iter)

token_embeddings=token_embedding_layeer(inputs)

context_size=max_length

pos_embedding_layer=torch.nn.Embedding(context_size,output_dim)

pos_embedding=pos_embedding_layer(torch.arange(context_size))

input_embeddings=token_embeddings+pos_embedding
print(input_embeddings)



        

    