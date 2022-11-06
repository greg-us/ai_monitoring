import re
import os
import torch.nn.functional as F
import torch
import random
import time
import numpy as np
import spacy
from tqdm import tqdm
from matplotlib import pyplot as plt

import Model

def loadDataset(fileNames):
    print('--Loading files')
    fileContent = []

    for path in fileNames:
        if os.path.isfile(path) :
            with open(path, "r") as f:
                fileContent = fileContent + f.readlines()

    # Specific for log file
    for i in range(len(fileContent)):
        if fileContent[i].find('[') > -1 and fileContent[i].find(']') > -1:
            dateStr = fileContent[i][fileContent[i].find('[')
                                     +1:fileContent[i].find(']')]
            fileContent[i] = fileContent[i][fileContent[i].find(']')+1:]
        else:
            dateStr = None

    print('--Remove duplicates in ', len(fileContent), ' lines')
    return list(set(fileContent))

# Initializing Dataset
#For retraining
filesToRead = ['data/standard.txt', 'data/dataset.txt']
#For initial training
#filesToRead = ['logs/error.log', 'logs/access.log']
print('-Loading Dataset')
dataset = loadDataset(filesToRead)
print('-Loaded ', len(dataset), ' items')

# Initializing Spacy
try:
    tokenizer = spacy.load('en_core_web_md')
except:
    os.system('python -m spacy download en_core_web_md')
    tokenizer = spacy.load('en_core_web_md')

history = []
ds = []

print('-Replacement')
for i in range(len(dataset)) :
    ds.append(re.sub('[0-9]', '', dataset[i]))

print('-Deduplication')
written = []
for i in range(len(ds)-1,0,-1) :
    if not ds[i] in written:
        written.append(ds[i])
    else :
        dataset.pop(i)

print('-Tokenizable lines : ', len(dataset))
with open("data/dataset.txt", "w") as ds:
    ds.writelines(dataset)

## Take extra time to tokenize line by lines
## and check if dataset is different
## If too long, this step can be avoided
maxLenCompare = Model.TOKEN_LENGTH
ds = []

for i, sample in enumerate(tqdm(dataset)):
    sentence = tokenizer(sample)
    temp = list(sentence[:maxLenCompare])
    temp = ' '.join(str(x) for x in temp)

    if not temp in history :
        history.append(temp)
        ds.append(sample)

with open("data/dataset.txt", "w") as dsf:
    dsf.writelines(ds)
