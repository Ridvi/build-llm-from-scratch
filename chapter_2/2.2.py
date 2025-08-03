import re
with open('chapter_2/2.2.txt', 'r') as file:
    text=file.read()

result=text.split(' ')
all_words = sorted(set(result))

vocab={token: i for i, token in enumerate(all_words)}

for items in vocab.items():
    print(items)


class SimpleTokenizer:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self,text):
        result= re.split(r'([,.?"!()\'] | -- |\s)',text)
        result = [item.strip() for item in result if item.strip()]

        ids = [self.str_to_int[i] for i in result]

        return ids
    
    def decode(self,ids):
        text=' '.join([self.int_to_str[i] for i in ids ])
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text


tokenizer = SimpleTokenizer(vocab)
ids=tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

