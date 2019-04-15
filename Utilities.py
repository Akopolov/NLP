import os
import random
import unicodedata
import string

def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

data = []
path = 'names/'
entries = os.listdir(path)

for entrie in entries:
    language = entrie.split('.')[0]
    file = open(path + entrie, "r")
    for name in file:
        data.append(unicodeToAscii(name.rstrip()) + "###" + language + "\n")

random.shuffle(data, random.random)

df = {
    "Train" : [],
    "Validate" : [],
    "Test" : []
}

traine_size = int(len(data) * 0.75)

df["Train"] = data[:traine_size]
temp = data[traine_size:]

validation_size = int(len(temp) * 0.9)
df["Validate"] = temp[:validation_size]
df["Test"] = temp[validation_size:]

for k in df.keys():
    file = open('data/' + k + ".txt", "w")
    
    for line in df[k]:
        file.write(line)
    
    file.close()
