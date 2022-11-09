import torch
import random
import numpy as np
import spacy
import os
import sys
from matplotlib import pyplot as plt
from Model import Autoencoder, TOKEN_LENGTH, EMBEDDING_SIZE


def batch(tokenizer, dataset, batchsize=32, embedding_size=256):
    samples = []

    for _i in range(batchsize):
        samples.append(dataset[0])
        dataset.append(dataset.pop(0))

    embeddings = []
    for i, sample in enumerate(samples):
        sentence = tokenizer(sample)
        embeddings.append([])
        for y, token in enumerate(sentence):
            if y < TOKEN_LENGTH:
                embeddings[i].append(np.array(token.vector[:embedding_size]))
        for _z in range(TOKEN_LENGTH-len(embeddings[i])):
            embeddings[i].append(np.zeros(embedding_size))

    embeddings = np.array(embeddings)

    return torch.from_numpy(
                embeddings.reshape(
                    [batchsize, 1, TOKEN_LENGTH, embedding_size]
                )
           ).double()


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# Initializing Autoencoder
print('-Create Model')
model = Autoencoder()
model = model.double()
print('-Load')

# Check previous steps
assert os.path.isfile('model/AE.pt'), "model weights missing"
assert os.path.isfile('data/dataset.txt'), "dataset missing"
device = get_device()
try:
    if device == 'cpu':
        model.load_state_dict(
            torch.load('model/AE.pt', map_location=torch.device('cpu'))
        )
    else:
        model.load_state_dict(torch.load('model/AE.pt'))
except Exception:
    print("Can't load checkpoint. Abort.")
    exit(1)

# Initializing Spacy
tokenizer = spacy.load('en_core_web_md')
# Load dataset
dataset = open("data/dataset.txt", "r").readlines()
random.shuffle(dataset)
outliers = open("data/outliers.txt", "r").readlines()
print('-Loaded ', len(dataset), ' items')

# Loss function
criterion = torch.nn.MSELoss()

print('-Execute on ', device)
model.to(device)

print('-Validating')

# Validate loss
val_size = len(dataset)/10
val_loss = 0.0
samples = batch(tokenizer, dataset, val_size, EMBEDDING_SIZE)
samples = samples.to(device)

outputs = model(samples)
loss = criterion(outputs, torch.tensor(samples).clone().detach())
val_loss = loss.item()*samples.size(0)

print('Average Loss: {:.6f}'.format(val_loss/val_size))

samples = samples.detach().cpu().numpy()
outputs = outputs.detach().cpu().numpy()
mses = np.array([np.mean(np.power(samples[x][0]
                - outputs[x][0], 2)) for x in range(len(samples))])
highest = max(mses)
print('Highest MSE: {:.6f}'.format(highest))

percentile = np.percentile(mses, 99)
print('99%tile: {:.6f}'.format(percentile))

# Validate one-by-one
val_samples_size = 3
samples = batch(tokenizer, dataset, val_samples_size, EMBEDDING_SIZE)
samples = samples.to(device)
outputs = model(samples)

samples = samples.detach().cpu().numpy()
outputs = outputs.detach().cpu().numpy()

# Plot vectors comparison
for i in range(len(samples)):
    mse = np.mean(np.power(samples[i][0] - outputs[i][0], 2))
    plt.figure(figsize=(20, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Loss: {:.6f}'.format(mse))
    ax1.set_title('In')
    ax1.set_axis_off()
    ax1.imshow(samples[i][0])
    ax2.set_title('Out')
    ax2.set_axis_off()
    ax2.imshow(outputs[i][0])
    fig.savefig(f'./out_{i}.png'.format(i), bbox_inches='tight',
                pad_inches=0, dpi=300)

# Test outliers
samples = batch(tokenizer, outliers, len(outliers), EMBEDDING_SIZE)
samples = samples.to(device)

outputs = model(samples)
loss = criterion(outputs, torch.tensor(samples).clone().detach())
val_loss = loss.item()*samples.size(0)

print('Average Outlier Loss: {:.6f}'.format(val_loss/len(outliers)))

samples = samples.detach().cpu().numpy()
outputs = outputs.detach().cpu().numpy()
lowest = min([np.mean(np.power(samples[x][0] - outputs[x][0], 2))
             for x in range(len(samples))])

print('Lowest MSE: {:.6f}'.format(lowest))

with open("model/threshold.dat", "w") as thresh:
    if highest*1.1 < lowest:
        print('Highest mse on dataset is lower than lowest mse for outliers :')
        print(' => Possible thershold : {:.6f}'
              .format(highest+(lowest-highest)/2))
        thresh.writelines(str(highest+(lowest-highest)/2))
        sys.exit(0)
    elif percentile*2 < lowest:
        print('Difference between 99%tile and \
              lowest mse for outlier is sufficient :')
        print(' => Possible thershold : {:.6f}'
              .format(percentile*1.5+(lowest-percentile*1.5)/2))
        thresh.writelines(str(percentile*1.5+(lowest-percentile*1.5)/2))
        sys.exit(0)

sys.exit(2)
