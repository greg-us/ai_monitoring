import torch.nn.functional as F
import torch

#To keep NN simple, embedding size and token counts are limited.
#Token count Limit can be set higher than mean length
TOKEN_LENGTH = 64
#Embedding size from spacy is 256, cutting it shows little impact on performances
EMBEDDING_SIZE = 64

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        #latent space
        self.dense1 = torch.nn.Linear(TOKEN_LENGTH*EMBEDDING_SIZE, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.dense3 = torch.nn.Linear(256, 256)
        self.dense4 = torch.nn.Linear(256, TOKEN_LENGTH*EMBEDDING_SIZE)

    def forward(self, x):
        batch, channels, width, height = x.shape
        x = x.reshape(batch, -1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        x = x.view(batch, channels, width, height)

        return x
