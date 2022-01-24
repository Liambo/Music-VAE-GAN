import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import os
import pypianoroll


def load_batch(genre, batch_size):
    directory = os.getcwd() + '/Dataset/Genres/' + genre
    batch = []
    try:
        for file in os.listdir(directory):
            pianoroll = pypianoroll.load(directory + '/' + file)
            tempbatch = []
            for track in pianoroll: # Loading tracks into a list
                if track.pianoroll.size == 0: # If track is empty, append None
                    tempbatch.append(None)
                else:
                    tempbatch.append(track.pianoroll[:, 20:104])
                    empty = np.zeros_like(track.pianoroll[:, 20:104]) # For replacing empty tracks later
            for i in range(len(tempbatch)): # Replacing empty tracks with 0-arrays so all tracks have same shape
                if tempbatch[i] is None:
                    tempbatch[i] = empty
            tempbatch = np.array(tempbatch) # Convert batch to numpy array

            # Concatenate all separate track vectors into a single vector
            tempbatch = np.concatenate((tempbatch[0], tempbatch[1], tempbatch[2],tempbatch[3], tempbatch[4]), axis = 1)
            for i in range(tempbatch.shape[0]//96): # Split song up into 4 beat segments i.e. 1 bar 
                batch.append(np.array(tempbatch[96*i:96*(i+1), :]))
                if len(batch) >= batch_size:
                    break
            print('loaded {} of {}'.format(len(batch), batch_size))
            if len(batch) >= batch_size:
                break
    except FileNotFoundError:
        print('Error: no such genre as', genre)
        return
    return np.array(batch)


class VAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, latent_dim,
    batch_first=True):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            #Define layers
            self.encoder = nn.GRU(input_size=self.input_size, 
            hidden_size=self.hidden_dim)
            self.decoder = nn.GRU(input_size = self.input_size,
            hidden_size=self.hidden_dim)
            self.fc_mu = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim)
            self.fc_var = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim)
            self.fc_output = nn.Linear(in_features=self.latent_dim,
            out_features=self.hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def forward(self, x):
        _, x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        var = self.fc_var(x_encoded)
        z = self.reparameterize(mu, var)
        hidden = self.fc_output(z)
        x_hat, _ = self.decoder(hidden)
        return x_hat

batch = load_batch('Jazz', 64)
input_tensor = torch.from_numpy(batch)
input_tensor = input_tensor.type(torch.FloatTensor)
vae = VAE(420, 420, 128, 128)
print(vae.forward(input_tensor).shape)