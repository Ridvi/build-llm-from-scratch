with open('2.2.txt', 'r') as file:
    text=file.read()

result=text.split(' ')
all_words = sorted(set(result))

vocab={token: i for i, token in enumerate(all_words)}
for items in vocab.items():
    print(items)

