from statistics import mean
import torch
from torch import nn
import numpy as np

# consider bidirectional, start notes have more importance on 2nd pass

class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim, n_genres=5):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.n_genres = n_genres

            #Define layers
            self.encoder = nn.GRUCell(input_size=self.input_size+self.n_genres, 
            hidden_size=self.hidden_dim)
            self.decoder = nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)
            self.discriminator = nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)
            self.fc_mu = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim-self.n_genres)
            self.fc_logvar = nn.Linear(in_features=self.hidden_dim,
            out_features=self.latent_dim-self.n_genres)
            self.fc_output = nn.Linear(in_features=self.latent_dim,
            out_features=self.hidden_dim)
            self.fc_decoder = nn.Linear(in_features=self.hidden_dim,
            out_features=self.input_size)
            self.fc_discriminator = nn.Linear(in_features=self.hidden_dim,
            out_features=2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def forward(self, x, latent_genre_vec): # Forward for TRANSFER not GENERATION
        hidden = torch.zeros((x.shape[1], self.hidden_dim)) # Initialise hidden state as zeroes
        for i in range(x.shape[0]): # For i in range(len of piece)
            hidden = self.encoder(x[i], hidden) # Update hidden state for every timestep
        mu = self.fc_mu(hidden) # Get latent mean from final hidden state
        logvar = self.fc_logvar(hidden) # Get latent logvar from final hidden state
        z = self.reparameterize(mu, logvar) # Reparamaterize & sample from latent space
        z = torch.concat((z, latent_genre_vec), axis=1)
        hidden = self.fc_output(z) # Sample from latent space to get initial hidden state for decoder
        output_note = torch.zeros((x.shape[1], self.input_size)) # Note initialised as empty, SHOULD BE PREV NOTE IF NOT NEW SONG OR START TOKEN IF NEW SONG
        output = torch.zeros(0, x.shape[1], self.input_size)
        for i in range(x.shape[0]): # For style transfer: NEEDS SOFTMAX FNCT FOR GENRE TRANSFER
            hidden = self.decoder(output_note, hidden)
            output_note = self.fc_decoder(hidden)
            output = torch.cat((output, torch.unsqueeze(output_note, 0)), 0)
        return output, mu, logvar


class Optimisation:
    def __init__(self, model, optimiser, beta, genre_dict, device):
        self.device=device
        self.model = model
        self.optimiser = optimiser
        self.train_losses = []
        self.val_losses = []
        self.mse = nn.MSELoss().to(self.device)
        self.beta = beta
        self.genre_dict = genre_dict
        self.n_genres = len(genre_dict)

    def loss_fn(self, x, x_hat, mu, logvar):
        mse_loss = self.mse(x[:,:,:-self.n_genres], x_hat) # reconstruction loss between input & output, i.e. how similar are they
        kl_div = (0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1).sum(axis=1)).mean()
        # KL Div loss, i.e. how similar is prior to posterior distribution. We sum loss over latent variables for each batch & timestep, then mean.
        
        return mse_loss + self.beta * kl_div
    
    def train_step(self, x, latent_genre_vec, vae=True):
        self.optimiser.zero_grad()
        self.model.train()
        if vae:
            x_hat, mu, logvar = self.model(x, latent_genre_vec)
            loss = self.loss_fn(x, x_hat, mu, logvar)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
    def test_step(self, x, latent_genre_vec):
        x_hat, mu, logvar = self.model(x, latent_genre_vec)
        loss = self.loss_fn(x, x_hat, mu, logvar)
        return loss.item()
    
    def train(self, train_set, test_set, n_epochs=50, eval_every=5):
        genre_vec_dict = {} # Dictionary to hold genre vectors for different genres
        latent_genre_vec_dict = {} # Dictionary to hold latent genre vectors for different genres
        for genre in self.genre_dict:
            genre_vec_dict[genre] = np.concatenate((np.zeros((train_set[0].shape[0], train_set[0].shape[1], self.genre_dict[genre])), np.ones((train_set[0].shape[0], train_set[0].shape[1], 1)), np.zeros((train_set[0].shape[0], train_set[0].shape[1], self.n_genres-(1+self.genre_dict[genre])))), axis=2)
            # Constructs genre vector of 0's with a 1 at position of genre from "genre_dict". This allows conditional encoding for CVAE by concatenating genre info onto end of inputs. We create this vector once at the start, rather than for every batch.
            latent_genre_vec_dict[genre] = torch.concat((torch.zeros((train_set[0].shape[1], self.genre_dict[genre])), torch.ones((train_set[0].shape[1], 1)), torch.zeros((train_set[0].shape[1], self.n_genres-(1+self.genre_dict[genre])))), axis=1).to(device=self.device)
            # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        for epoch in range(1, n_epochs+1):
            for batch in train_set:
                genre = batch[0]
                data = batch[1]
                input = torch.from_numpy(np.concatenate((data, genre_vec_dict[genre]), axis=2).astype('float32')).to(device=self.device)
                # Constructs input by concatenating genre vector to end of every input vector & converting entire matrix to float32 for compatability.
                self.train_step(input, latent_genre_vec_dict[genre])
                print('done batch')
            print('done epoch {} of {}'.format(epoch, n_epochs))
            if epoch % eval_every == 0:
                eval_losses = []
                for batch in test_set:
                    genre = batch[0]
                    data = batch[1]
                    input = torch.from_numpy(data).to(device=self.device)
                    loss = self.test_step(input, latent_genre_vec_dict[genre])
                    eval_losses.append(loss)
                print('losses for epoch', epoch)
                print(eval_losses)
                print('mean loss: ' + mean(eval_losses))

