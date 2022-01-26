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
            songroll = []
            for track in pianoroll: # Loading tracks into a list
                if track.pianoroll.size == 0: # If track is empty, append None
                    songroll.append(None)
                else:
                    songroll.append(track.pianoroll[:, 20:104])
                    empty = np.zeros_like(track.pianoroll[:, 20:104]) # For replacing empty tracks later
            for i in range(len(songroll)): # Replacing empty tracks with 0-arrays so all tracks have same shape
                if songroll[i] is None:
                    songroll[i] = empty
            songroll = np.array(songroll) # Convert batch to numpy array
            # Concatenate all separate track vectors into a single vector
            songroll = np.concatenate((songroll[0], songroll[1], songroll[2],songroll[3], songroll[4]), axis = 1)
            for i in range(songroll.shape[0]//96): # Split song up into 4 beat segments i.e. 1 bar 
                batch.append(np.array(songroll[96*i:96*(i+1), :]))
                if len(batch) >= batch_size:
                    break
            print('loaded {} of {}'.format(len(batch), batch_size))
            if len(batch) >= batch_size:
                break
    except FileNotFoundError:
        print('Error: no such genre as', genre)
        return
    
    return np.transpose(np.array(batch), (1, 0, 2)) # Tranpose so batch is 2nd dimension DECIDE IF WORKS

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.GRU = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_dim)


class VAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, latent_dim,
    batch_first=False):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            #Define layers
            self.encoder = nn.GRUCell(input_size=self.input_size, 
            hidden_size=self.hidden_dim)
            self.GRU1 = nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)
            self.GRU2 = nn.GRUCell(input_size=self.hidden_dim,
            hidden_size=self.input_size)
            self.fc_mu = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim)
            self.fc_var = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim)
            self.fc_output = nn.Linear(in_features=self.latent_dim,
            out_features=self.hidden_dim)
            self.fc_decoder = nn.Linear(in_features=self.hidden_dim,
            out_features=420)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def forward(self, x): # Forward for TRANSFER not GENERATION
        hidden = torch.zeros((x.shape[1], self.hidden_dim)) # Initialise hidden state as zeroes
        print(hidden.shape)
        for i in range(x.shape[0]): # Batch-first, so for i in range(len of piece)
            print(x[i].shape)
            hidden = self.encoder(x[i], hidden) # Update hidden state for every timestep
        mu = self.fc_mu(hidden) # Get latent mean from final hidden state
        var = self.fc_var(hidden) # Get latent logvar from final hidden state
        z = self.reparameterize(mu, var) # Reparamaterize & sample from latent space
        hidden = self.fc_output(z) # Sample from latent space to get initial hidden state for decoder
        output_note = torch.zeros((x.shape[1], x.shape[2])) # Note initialised as empty, SHOULD BE PREV NOTE IF NOT NEW SONG OR START TOKEN IF NEW SONG
        output = torch.zeros(0, x.shape[1], x.shape[2])
        for i in range(x.shape[0]): # For style transfer: NEEDS SOFTMAX FNCT FOR GENRE TRANSFER
            hidden = self.GRU1(output_note, hidden)
            output_note = self.fc_decoder(hidden)
            output = torch.cat((output, torch.unsqueeze(output_note, 0)), 0)
        return output

batch = load_batch('Jazz', 64)
input_tensor = torch.from_numpy(batch)
input_tensor = input_tensor.type(torch.FloatTensor)
print(input_tensor.shape)
vae = VAE(420, 420, 128, 128)
print(vae.forward(input_tensor).shape)