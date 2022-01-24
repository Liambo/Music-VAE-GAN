from turtle import forward
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import os
import pypianoroll
#np.set_printoptions(threshold=np.inf)


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
                    tempbatch.append(track.pianoroll)
                    empty = np.zeros_like(track.pianoroll) # For replacing empty tracks later
            for i in range(len(tempbatch)): # Replacing empty tracks with 0-arrays so all tracks have same shape
                if tempbatch[i] is None:
                    tempbatch[i] = empty
            songroll = np.array(tempbatch)
            for i in range(songroll.shape[1]//96): # Split song up into 4 beat segments i.e. 1 bar 
                batch.append(songroll[:, 96*i:96*(i+1), :])
                if len(batch) >= batch_size:
                    break
            print('loaded {} of {}'.format(len(batch), batch_size))
            if len(batch) >= batch_size:
                break
    except FileNotFoundError:
        print('Error: no such genre as', genre)
        return
    return batch

class VAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, latent_dim,
    n_layers, n_tracks, drop_prob=0.2):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.n_layers = n_layers
            self.n_tracks = n_tracks
            self.drop_prob = drop_prob

            #Define layers
            self.rnn = nn.GRU(input_size=self.input_size, 
            hidden_size=self.hidden_dim, dropout=self.drop_prob)
            self.fc1 = nn.Linear(input_size=self.hidden_dim*5,
            output_size=self.latent_dim*2)
            self.relu1= nn.ReLU()
            self.fc2 = nn.Linear(input_size=self.latent_dim,
            output_size=self.latent_dim)
            self.relu2 = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def forward(self, x):
        x = [self.rnn(a) for a in x]
        return x

batch = load_batch('Jozz', 1024)


    

            

