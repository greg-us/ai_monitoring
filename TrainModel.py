import torch
import random
import numpy as np
import spacy
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

device = get_device()
try:
    if device == 'cpu':
        model.load_state_dict(
            torch.load('model/AE.pt', map_location=torch.device('cpu'))
        )
    else:
        model.load_state_dict(torch.load('model/AE.pt'))
except Exception:
    print("Can't load checkpoint. Random init.")

# Initializing Spacy
tokenizer = spacy.load('en_core_web_md')

# Initializing Dataset
print('-Loading Dataset')
dataset = open("data/dataset.txt", "r").readlines()
random.shuffle(dataset)
print('-Loaded ', len(dataset), ' items')

# Loss function
criterion = torch.nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

print('-Training on ', device)
model.to(device)


# Epochs
batch_size = 32
n_epochs = 5
steps = int(len(dataset)/batch_size*n_epochs)
print('-Training')

for step in range(steps):
    # monitor training loss
    train_loss = 0.0

    # Training
    samples = batch(tokenizer, dataset, batch_size, EMBEDDING_SIZE)
    samples = samples.to(device)
    optimizer.zero_grad()
    outputs = model(samples)

    loss = criterion(outputs, torch.tensor(samples).clone().detach())
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*samples.size(0)

    print('Step: {} / {} \tTraining Loss: {:.6f}'
          .format(step, steps, train_loss))

    if step % 10 == 0:
        torch.save(model.state_dict(), 'model/AE.pt')
