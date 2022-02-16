from statistics import mean
import torch
from torch import nn
import numpy as np
import datetime

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
        hidden = torch.zeros((x.shape[0], self.hidden_dim)) # Initialise hidden state as zeroes
        for i in range(x.shape[1]): # For i in range(len of piece)
            hidden = self.encoder(x[:, i].squeeze(), hidden) # Update hidden state for every timestep
        mu = self.fc_mu(hidden) # Get latent mean from final hidden state
        logvar = self.fc_logvar(hidden) # Get latent logvar from final hidden state
        z = self.reparameterize(mu, logvar) # Reparamaterize & sample from latent space
        z = torch.concat((z, latent_genre_vec), axis=1)
        hidden = self.fc_output(z) # Sample from latent space to get initial hidden state for decoder
        output_note = torch.zeros((x.shape[0], self.input_size)) # Note initialised as empty, SHOULD BE PREV NOTE IF NOT NEW SONG OR START TOKEN IF NEW SONG. Just start from first note of input?
        output = torch.zeros((x.shape[0], 0, self.input_size))
        for i in range(x.shape[1]):
            hidden = self.decoder(output_note, hidden)
            output_note = self.fc_decoder(hidden)
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output, mu, logvar


class Optimisation:
    def __init__(self, model, checkpoint_path, optimiser, beta, genre_dict, batch_size, device):
        self.device=device
        self.model = model
        self.optimiser = optimiser
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.val_losses = []
        self.mse = nn.MSELoss().to(self.device)
        self.beta = beta
        self.batch_size = batch_size
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
    
    def train(self, train_loader, test_loader, n_epochs, eval_every):
        genre_vec_dict = {} # Dictionary to hold genre vectors for different genres
        latent_genre_vec_dict = {} # Dictionary to hold latent genre vectors for different genres
        for genre in self.genre_dict:
            genre_vec_dict[genre] = torch.concat((torch.zeros(96, self.genre_dict[genre]), torch.ones(96, 1), torch.zeros(96, self.n_genres-(1+self.genre_dict[genre]))), axis=1)
            # Constructs genre vector of 0's with a 1 at position of genre from "genre_dict". This allows conditional encoding for CVAE by concatenating genre info onto end of inputs. We create this vector once at the start, rather than for every batch.
            latent_genre_vec_dict[genre] = torch.concat((torch.zeros(self.genre_dict[genre]), torch.ones(1), torch.zeros(self.n_genres-(1+self.genre_dict[genre]))))
            # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        for epoch in range(1, n_epochs+1):
            for batch, genre in train_loader:
                input = torch.concat((batch, torch.stack([genre_vec_dict[gen] for gen in genre])), axis=2).to(self.device)
                # Constructs input by concatenating genre vector to end of every input vector & converting entire matrix to float32 for compatability.
                self.train_step(input, torch.stack([latent_genre_vec_dict[gen] for gen in genre]).to(self.device))
                print('done batch')
            print('done epoch {} of {}'.format(epoch, n_epochs))
            if epoch % eval_every == 0:
                eval_losses = []
                for batch, genre in test_loader:
                    input = torch.concat((batch, torch.stack([genre_vec_dict[gen] for gen in genre])), axis=2).to(self.device)
                    loss = self.test_step(input, torch.stack([latent_genre_vec_dict[gen] for gen in genre]).to(self.device))
                    eval_losses.append(loss)
                print('mean loss for epoch ' + epoch + ': ' + mean(eval_losses))
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimiser_state_dict': self.optimiser.state_dict(), 'loss': mean(eval_losses)}, self.checkpoint_path + str(datetime.datetime()) + '.pt')



