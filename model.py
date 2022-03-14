from statistics import mean
import torch
from torch import nn
import numpy as np
import datetime
import time


class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim,
                    bidirectional, fc_layers, device, n_genres=5):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.bidirectional = bidirectional
            self.fc_layers = fc_layers
            self.n_genres = n_genres
            self.device = device
            if bidirectional: # If bidirectional, hidden dim for producing latent space has to be twice as large due to concatenation of encoder outputs.
                self.encoder_hidden_dim = 2*hidden_dim
            else:
                self.encoder_hidden_dim = hidden_dim
            #Define layers
            self.encoder = nn.GRUCell(input_size=self.input_size+self.n_genres, # Genre vector is concatenated to input to encoder, hence input_size + n_genres
            hidden_size=self.hidden_dim)
            if bidirectional:
                self.back_encoder = nn.GRUCell(input_size=self.input_size+self.n_genres, 
                hidden_size=self.hidden_dim)
                self.back_discriminator = nn.GRUCell(input_size=self.input_size, 
                hidden_size=self.hidden_dim)
            self.decoder = nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)
            self.discriminator = nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)
            self.fc_mu = nn.ModuleList([])
            self.fc_logvar = nn.ModuleList([])
            self.fc_latent = nn.ModuleList([])
            self.fc_decoder = nn.ModuleList([])
            self.fc_discriminator = nn.ModuleList([])

            for _ in range(fc_layers-1):
                self.fc_mu.append(nn.Sequential(
                    nn.Linear(in_features=self.encoder_hidden_dim,
                    out_features=self.encoder_hidden_dim),
                    nn.LeakyReLU()
                ))
                self.fc_logvar.append(nn.Sequential(
                    nn.Linear(in_features=self.encoder_hidden_dim,
                    out_features=self.encoder_hidden_dim),
                    nn.LeakyReLU()
                ))
                self.fc_latent.append(nn.Sequential(
                    nn.Linear(in_features=self.latent_dim,
                    out_features=self.latent_dim),
                    nn.LeakyReLU()
                ))
                self.fc_decoder.append(nn.Sequential(
                    nn.Linear(in_features=self.hidden_dim,
                    out_features=self.hidden_dim),
                    nn.LeakyReLU()
                ))
                self.fc_discriminator.append(nn.Sequential(
                    nn.Linear(in_features=self.encoder_hidden_dim,
                    out_features=self.encoder_hidden_dim),
                    nn.LeakyReLU()
                ))

            self.fc_mu.append(nn.Linear(in_features=self.encoder_hidden_dim,
            out_features=self.latent_dim-self.n_genres))
            self.fc_logvar.append(nn.Linear(in_features=self.encoder_hidden_dim,
            out_features=self.latent_dim-self.n_genres))
            self.fc_latent.append(nn.Linear(in_features=self.latent_dim,
            out_features=self.hidden_dim))
            self.fc_decoder.append(nn.Linear(in_features=self.hidden_dim,
            out_features=self.input_size))
            self.fc_discriminator.append(nn.Sequential(
                nn.Linear(in_features=self.encoder_hidden_dim, out_features=1), nn.Sigmoid()))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def transfer(self, x, latent_genre_vec): # Transfer x to target genre in latent_genre_vec
        hidden = torch.zeros((x.shape[0], self.hidden_dim), device=self.device) # Initialise hidden state as zeroes
        for i in range(x.shape[1]): # For i in range(len of piece)
            hidden = self.encoder(x[:, i].squeeze(), hidden) # Update hidden state for every timestep
        if self.bidirectional:
            back_hidden = torch.zeros((x.shape[0], self.hidden_dim), device=self.device)
            for i in range(x.shape[1]-1, -1, -1):
                back_hidden = self.back_encoder(x[:, i].squeeze(), back_hidden)
            hidden = torch.cat((hidden, back_hidden), dim=1)
        mu = self.fc_mu[0](hidden) # Get latent mean from final hidden state
        logvar = self.fc_logvar[0](hidden) # Get latent logvar from final hidden state
        for i in range(self.fc_layers-1):
            mu = self.fc_mu[i+1](mu)
            logvar = self.fc_logvar[i+1](logvar)
        z = self.reparameterize(mu, logvar) # Reparamaterize & sample from latent space
        z = torch.cat((z, latent_genre_vec), axis=1)
        hidden = self.fc_latent[0](z) # Sample from latent space to get initial hidden state for decoder
        for i in range(self.fc_layers-1):
            hidden = self.fc_latent[i+1](hidden)
        output_note = x[:, :1, :self.input_size].squeeze() # Start from first note of input (so not starting from empty note)
        output = x[:, :1, :self.input_size] # We only take up to input_size as genre info not concatenated to output.
        for i in range(x.shape[1]-1):
            hidden = self.decoder(output_note, hidden)
            output_note = self.fc_decoder[0](hidden)
            for i in range(self.fc_layers-1):
                output_note = self.fc_decoder[i+1](hidden)
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output, mu, logvar
    
    def generate(self, x, latent_genre_vec): # Generate a piece in style of genre in latent_genre_vec
        z = torch.cat((torch.randn((latent_genre_vec.shape[0], self.latent_dim-self.n_genres), device=self.device), latent_genre_vec), dim=1)
        hidden = self.fc_latent[0](z) # Sample from latent space to get initial hidden state for decoder
        for i in range(self.fc_layers-1):
            hidden = self.fc_latent[i+1](hidden)
        output_note = x[:, :1, :self.input_size].squeeze() # Start from first note of input (so not starting from empty note)
        output = x[:, :1, :self.input_size]
        for i in range(x.shape[1]-1):
            hidden = self.decoder(output_note, hidden)
            output_note = self.fc_decoder[0](hidden)
            for i in range(self.fc_layers-1):
                output_note = self.fc_decoder[i+1](hidden)
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output
    
    def discriminate(self, x):
        hidden = torch.zeros((x.shape[0], self.hidden_dim), device=self.device) # Initialise hidden state as zeroes
        for i in range(x.shape[1]): # For i in range(len of piece)
            hidden = self.discriminator(x[:, i, :].squeeze(), hidden) # Update hidden state for every timestep
        if self.bidirectional:
            back_hidden = torch.zeros((x.shape[0], self.hidden_dim), device=self.device)
            for i in range(x.shape[1]-1, -1, -1):
                back_hidden = self.back_discriminator(x[:, i, :].squeeze(), back_hidden)
            hidden = torch.cat((hidden, back_hidden), dim=1)
        hidden = self.fc_discriminator[0](hidden)
        for i in range(self.fc_layers-1):
            hidden = self.fc_discriminator[i+1](hidden)
        return hidden


class Optimisation:
    def __init__(self, model, checkpoint_path, optimiser, beta, genre_dict, batch_size, vae_mse, device):
        self.device=device
        self.model = model
        self.optimiser = optimiser
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.val_losses = []
        self.vae_mse = vae_mse
        self.mse = nn.MSELoss().to(self.device)
        self.bce = nn.BCELoss().to(self.device)
        self.beta = beta
        self.batch_size = batch_size
        self.genre_dict = genre_dict
        self.n_genres = len(genre_dict)

    def vae_loss_fn(self, x, x_hat, mu, logvar):
        if self.vae_mse == True:
            loss = self.mse(x[:,:,:-self.n_genres], x_hat) # reconstruction loss between input & output, i.e. how similar are they
        else:
            loss = self.bce(x_hat, x[:,:,:-self.n_genres])
        kl_div = (0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1).sum(axis=1)).mean()
        # KL Div loss, i.e. how similar is prior to posterior distribution. We sum loss over latent variables for each batch & timestep, then mean.
        return loss + self.beta * kl_div
    
    def gan_loss_fn(self, x, x_hat):
        bce_loss = self.bce(x_hat, x)
        return bce_loss
    
    def measure_qn(self, x): # Measure 'qualified notes', i.e. notes longer than 3 timesteps
        pointer_tensor = torch.zeros((x.shape[0], x.shape[2]), device=self.device) # Pointer tensor keeps track of length of currently on notes
        qn_num = 0 # qualified notes
        uqn_num = 0 # unqualified notes
        for i in range(x.shape[1]+1):
            if i < x.shape[1]:
                current_slice = x[:, i, :].squeeze()
            else:
                current_slice = torch.zeros_like(current_slice)
            pointer_tensor += (current_slice > 64)
            qn = (((current_slice <= 64) * pointer_tensor) >= 3).sum()
            qn_num += qn
            uqn_num += ((((current_slice <= 64) * pointer_tensor) > 0)).sum() - qn
            pointer_tensor *= (current_slice > 64)
            # Note is considered on if value is above 64, so (current_slice > 64) will return a tensor of 0's and 1's, where 1 denotes a note that is on.
            # If current note is off, 'current_slice <= 64' = True, so multiply by pointer_tensor to get currently ending notes & their lengths.
            # If lengths are 3 or longer, that is a qualified note, so add to total qualified notes. If 1 or 2, it's unqualified, so add to total unqualified.
            # Multiply pointer by current notes at end so any finished notes get lengths set back to 0.
        return qn_num, uqn_num

    def measure_upc(self, x):
        pc_tracker = torch.zeros((x.shape[0], 60), device=self.device)
        n_notes = (x.shape[2]-2)//5
        for i in range(x.shape[1]):
            current_slice = x[:, i, :].squeeze()
            for j in range(n_notes // 12):
                pc_tracker += torch.cat((current_slice[:, j*12:min((j+1)*12, n_notes)],
                current_slice[:, n_notes + j*12:min(n_notes+(j+1)*12, 2*n_notes)],
                current_slice[:, 2*n_notes + j*12:min(2*n_notes+(j+1)*12, 3*n_notes)],
                current_slice[:, 3*n_notes + j*12:min(3*n_notes+(j+1)*12, 4*n_notes)],
                current_slice[:, 4*n_notes + j*12:min(4*n_notes+(j+1)*12, 5*n_notes)]))
        return [(pc_tracker[:, i*12:(i+1)*12] > 0).sum(1) for i in range(5)]

    def train_step(self, x, latent_genre_vec, vae=False, qn=True, upc=False):
        self.optimiser.zero_grad()
        self.model.train()
        if vae:
            x_hat, mu, logvar = self.model.transfer(x, latent_genre_vec)
            loss = self.vae_loss_fn(x, x_hat, mu, logvar)
        else:
            x_hat = self.model.generate(x, latent_genre_vec)
            fake_pred = self.model.discriminate(x_hat)
            fake_class = torch.ones_like(fake_pred)
            real_pred = self.model.discriminate(x[:, :, :-self.n_genres])
            real_class = torch.ones_like(real_pred)
            loss = self.gan_loss_fn(torch.cat((fake_pred, real_pred)), torch.cat((fake_class, real_class)))
        loss.backward()
        self.optimiser.step()
        if qn:
            qn_num, uqn_num = self.measure_qn(x_hat)
            if qn_num+uqn_num != 0:
                qn = qn_num/(qn_num+uqn_num)
            else:
                qn = 0
        return loss.item(), qn, upc
    
    def test_step(self, x, latent_genre_vec):
        x_hat, mu, logvar = self.model.transfer(x, latent_genre_vec)
        loss = self.vae_loss_fn(x, x_hat, mu, logvar)
        return loss.item()
    
    def train(self, train_loader, test_loader, writer, n_epochs, eval_every, measure_every, cycle_every, vae_train_proportion):
        genre_vec_dict = {} # Dictionary to hold genre vectors for different genres
        latent_genre_vec_dict = {} # Dictionary to hold latent genre vectors for different genres
        for genre in self.genre_dict:
            genre_vec_dict[genre] = torch.cat((torch.zeros(384, self.genre_dict[genre]), torch.ones(384, 1), torch.zeros(384, self.n_genres-(1+self.genre_dict[genre]))), axis=1)
            # Constructs genre vector of 0's with a 1 at position of genre from "genre_dict". This allows conditional encoding for CVAE by concatenating genre info onto end of inputs. We create this vector once at the start, rather than for every batch.
            latent_genre_vec_dict[genre] = torch.cat((torch.zeros(self.genre_dict[genre]), torch.ones(1), torch.zeros(self.n_genres-(1+self.genre_dict[genre]))))
            # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        running_loss = 0.0
        running_qn = 0.0
        tm1 = time.time()
        tm2 = time.time()
        ld = 0
        trn = 0
        for epoch in range(1, n_epochs+1):
            i=0
            for batch, genre in train_loader:
                tm2 = tm1
                tm1 = time.time()
                ld += tm1 - tm2
                input = torch.cat((batch, torch.stack([genre_vec_dict[gen] for gen in genre])), axis=2).to(self.device)
                # Constructs input by concatenating genre vector to end of every input vector & converting entire matrix to float32 for compatability.
                loss, qn, _ = self.train_step(input, torch.stack([latent_genre_vec_dict[gen] for gen in genre]).to(self.device), vae=vae)
                running_loss += loss
                running_qn += qn
                tm2 = tm1
                tm1 = time.time()
                trn += tm1 - tm2
                i += 1
                if i % measure_every == 0:
                    writer.writerow([epoch * len(train_loader) + i,
                                    running_loss / measure_every,
                                    running_qn / measure_every,
                                    ld, trn])
                    running_loss = 0.0
                    running_qn = 0.0
                    ld = 0
                    trn = 0
            if epoch == 1 or epoch % eval_every == 0:
                print('done epoch {} of {}'.format(epoch, n_epochs))
                eval_losses = []
                for batch, genre in test_loader:
                    input = torch.cat((batch, torch.stack([genre_vec_dict[gen] for gen in genre])), axis=2).to(self.device)
                    loss = self.test_step(input, torch.stack([latent_genre_vec_dict[gen] for gen in genre]).to(self.device))
                    eval_losses.append(loss)
                print('mean loss for epoch ' + str(epoch) + ': ' + str(mean(eval_losses)))
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimiser_state_dict': self.optimiser.state_dict(), 'loss': mean(eval_losses)}, self.checkpoint_path + str(datetime.datetime.now()) + '.pt')
            if epoch % cycle_every > cycle_every * vae_train_proportion:
                vae = False
            else:
                vae = True

    def lr_range_test(self, train_loader, writer, start_lr=-5.0, stop_lr=0.0, lr_step=0.5, measure_every=10, change_every=500):
        genre_vec_dict = {} # Dictionary to hold genre vectors for different genres
        latent_genre_vec_dict = {} # Dictionary to hold latent genre vectors for different genres
        for genre in self.genre_dict:
            genre_vec_dict[genre] = torch.cat((torch.zeros(384, self.genre_dict[genre]), torch.ones(384, 1), torch.zeros(384, self.n_genres-(1+self.genre_dict[genre]))), axis=1)
            # Constructs genre vector of 0's with a 1 at position of genre from "genre_dict". This allows conditional encoding for CVAE by concatenating genre info onto end of inputs. We create this vector once at the start, rather than for every batch.
            latent_genre_vec_dict[genre] = torch.cat((torch.zeros(self.genre_dict[genre]), torch.ones(1), torch.zeros(self.n_genres-(1+self.genre_dict[genre]))))
            # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        lr = 10**start_lr
        for g in self.optimiser.param_groups:
            g['lr'] = lr
        i = 0
        running_loss = 0.0
        while True:
            for batch, genre in train_loader:
                input = torch.cat((batch, torch.stack([genre_vec_dict[gen] for gen in genre])), axis=2).to(self.device)
                latent_genre_vec = torch.stack([latent_genre_vec_dict[gen] for gen in genre]).to(self.device)
                loss = self.train_step(input, latent_genre_vec)
                running_loss += loss
                i += 1
                if i % measure_every == 0:
                    writer.writerow([i, lr, running_loss / measure_every])
                    running_loss = 0.0
                if i % change_every == 0:
                    print('done test for lr ' + str(lr))
                    if start_lr >= stop_lr:
                        return
                    start_lr += lr_step
                    lr = 10**start_lr
                    for g in self.optimiser.param_groups:
                        g['lr'] = lr