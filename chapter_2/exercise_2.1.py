import tiktoken

tokenizer= tiktoken.get_encoding("gpt2")

text = 'Akwirw ier'
t=tokenizer.encode(text)
print(t)

print(tokenizer.decode(t))
