import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import os
import pypianoroll
import datetime


def load_batch(genre, batch_size, low_note, high_note):
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
                    songroll.append(track.pianoroll[:, low_note:high_note])
                    empty = np.zeros_like(track.pianoroll[:, low_note:high_note]) # For replacing empty tracks later
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
    
    return np.transpose(np.array(batch), (1, 0, 2)).astype('float32') # Tranpose so batch is 2nd dimension DECIDE IF WORKS


def convert_batch(batch, genre, path, low_note, high_note):
    filename = path + '/' + genre + '/' + str(datetime.datetime.now())
    note_range = high_note - low_note
    pianoroll = [[]] * 5
    print(pianoroll)
    batch = batch.tolist()
    for i in range(len(batch[0])):
        for j in range(len(batch)):
            for k in range(5):
                noteslist = batch[j][k][k*note_range:(k+1)*note_range]
                for l in range(len(noteslist)):
                    if noteslist[l] >= 0.5:
                        noteslist[l] = 1
                    else:
                        noteslist[l] = 0
                pianoroll[k].append([0]*low_note + noteslist + [0]*(128-high_note))
    return pianoroll   


class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim

            #Define layers
            self.encoder = nn.GRUCell(input_size=self.input_size, 
            hidden_size=self.hidden_dim)
            self.decoder = nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)
            self.fc_mu = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim)
            self.fc_logvar = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim)
            self.fc_output = nn.Linear(in_features=self.latent_dim,
            out_features=self.hidden_dim)
            self.fc_decoder = nn.Linear(in_features=self.hidden_dim,
            out_features=self.input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def forward(self, x): # Forward for TRANSFER not GENERATION
        hidden = torch.zeros((x.shape[1], self.hidden_dim)) # Initialise hidden state as zeroes
        for i in range(x.shape[0]): # For i in range(len of piece)
            hidden = self.encoder(x[i], hidden) # Update hidden state for every timestep
        mu = self.fc_mu(hidden) # Get latent mean from final hidden state
        logvar = self.fc_logvar(hidden) # Get latent logvar from final hidden state
        z = self.reparameterize(mu, logvar) # Reparamaterize & sample from latent space
        hidden = self.fc_output(z) # Sample from latent space to get initial hidden state for decoder
        output_note = torch.zeros((x.shape[1], x.shape[2])) # Note initialised as empty, SHOULD BE PREV NOTE IF NOT NEW SONG OR START TOKEN IF NEW SONG
        output = torch.zeros(0, x.shape[1], x.shape[2])
        for i in range(x.shape[0]): # For style transfer: NEEDS SOFTMAX FNCT FOR GENRE TRANSFER
            hidden = self.decoder(output_note, hidden)
            output_note = self.fc_decoder(hidden)
            output = torch.cat((output, torch.unsqueeze(output_note, 0)), 0)
        return output, mu, logvar


class Optimisation:
    def __init__(self, model, optimiser, beta):
        self.model = model
        self.optimiser = optimiser
        self.train_losses = []
        self.val_losses = []
        self.mse = nn.MSELoss()
        self.beta = beta

    def loss_fn(self, x, x_hat, mu, logvar):
        mse_loss = self.mse(x, x_hat) # reconstruction loss between input & output, i.e. how similar are they
        kl_div = (0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1).sum(axis=1)).mean()
        # KL Div loss, i.e. how similar is prior to posterior distribution. We sum loss over latent variables for each batch & timestep, then mean.
        return mse_loss + self.beta * kl_div
    
    def train_step(self, x):
        self.model.train()
        x_hat, mu, logvar = self.model(x)
        loss = self.loss_fn(x, x_hat, mu, logvar)
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
        return loss.item()
    
    def train(self, train_set, eval_set, n_epochs=50):
        train_losses = []
        for epoch in range(1, n_epochs+1):
            train_set = torch.from_numpy(load_batch('Jazz', 64, 20, 104))
            loss = self.train_step(train_set)
            train_losses.append(loss)
            print('done epoch {} of {}'.format(epoch, n_epochs))
        print(train_losses)

learning_rate = 0.001
weight_decay = 0.000001
batch = load_batch('Jazz', 64, 20, 104)
pianoroll = convert_batch(batch, 'none', 'none', 20, 104)
input_tensor = torch.from_numpy(batch)
input_tensor = input_tensor.type(torch.FloatTensor)
vae = VAE(420, 128, 128)
optimiser = Optimisation(vae, optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay), 1.0)
optimiser.train(None, None, 10)
output_tensor, _, _ = vae(input_tensor)
print(convert_batch(output_tensor, 'none', 'none', 20, 104))