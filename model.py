from statistics import mean
import torch
from torch import autograd, nn
import numpy as np
import datetime
import time
import math
from file_handling import convert_batch, convert_pianoroll
from copy import deepcopy


class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim, gru_layers,
                    fc_dropout, gru_dropout, bidirectional, fc_layers, device, n_genres=4):
            super(VAE, self).__init__()

            #Define parameters
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.gru_layers = gru_layers
            self.bidirectional = bidirectional
            self.fc_layers = fc_layers
            self.n_genres = n_genres
            self.fc_dropout = fc_dropout
            self.gru_dropout = nn.Dropout(p=gru_dropout)
            self.genre_classifier = nn.Softmax(dim=1)
            self.device = device
            if bidirectional: # If bidirectional, hidden dim for producing latent space has to be twice as large due to concatenation of encoder outputs.
                self.encoder_hidden_dim = 2*gru_layers*hidden_dim
            else:
                self.encoder_hidden_dim = gru_layers*hidden_dim

            #Define layers
            self.encoder = nn.ModuleList([nn.GRUCell(input_size=self.input_size, # Genre vector is concatenated to input to encoder, hence input_size + n_genres
                hidden_size=self.hidden_dim)])
            for i in range(self.gru_layers-1):
                self.encoder.append(nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))

            if bidirectional:
                self.back_encoder = nn.ModuleList([nn.GRUCell(input_size=self.input_size, 
                hidden_size=self.hidden_dim)])
                for i in range(self.gru_layers-1):
                    self.back_encoder.append(nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))

            self.decoder = nn.ModuleList([nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)])
            for i in range(self.gru_layers-1):
                self.decoder.append(nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))

            self.fc_mu = nn.Sequential()
            self.fc_logvar = nn.Sequential()
            self.fc_latent = nn.ModuleList([nn.Sequential()])
            self.fc_decoder = nn.Sequential()

            for i in range(self.fc_layers-1):
                self.fc_mu.add_module('Linear_' + str(i),
                    nn.Linear(in_features=self.encoder_hidden_dim,
                    out_features=self.encoder_hidden_dim))
                self.fc_mu.add_module('ReLU_' + str(i),
                    nn.LeakyReLU())
                self.fc_mu.add_module('Dropout_' + str(i),
                    nn.Dropout(p=self.fc_dropout))

                self.fc_logvar.add_module('Linear_' + str(i),
                    nn.Linear(in_features=self.encoder_hidden_dim,
                    out_features=self.encoder_hidden_dim))
                self.fc_logvar.add_module('ReLU_' + str(i),
                    nn.LeakyReLU())
                self.fc_logvar.add_module('Dropout_' + str(i),
                    nn.Dropout(p=self.fc_dropout))

                self.fc_latent[0].add_module('Linear_' + str(i),
                    nn.Linear(in_features=self.latent_dim,
                    out_features=self.latent_dim))
                self.fc_latent[0].add_module('ReLU_' + str(i),
                    nn.LeakyReLU())
                self.fc_latent[0].add_module('Dropout_' + str(i),
                    nn.Dropout(p=self.fc_dropout))
                
                self.fc_decoder.add_module('Linear_' + str(i),
                    nn.Linear(in_features=self.hidden_dim,
                    out_features=self.hidden_dim))
                self.fc_decoder.add_module('ReLU_' + str(i),
                    nn.LeakyReLU())
                self.fc_decoder.add_module('Dropout_' + str(i),
                    nn.Dropout(p=self.fc_dropout))

            self.fc_mu.add_module('Linear_' + str(self.fc_layers-1),
                nn.Linear(in_features=self.encoder_hidden_dim,
                out_features=self.latent_dim))

            self.fc_logvar.add_module('Linear_' + str(self.fc_layers-1),
                nn.Linear(in_features=self.encoder_hidden_dim,
                out_features=self.latent_dim))

            self.fc_latent[0].add_module('Linear_' + str(self.fc_layers-1),
                nn.Linear(in_features=self.latent_dim,
                out_features=self.hidden_dim))

            self.fc_decoder.add_module('Linear_' + str(self.fc_layers-1),
                nn.Linear(in_features=self.hidden_dim,
                out_features=self.input_size))
            self.fc_decoder.add_module('Sigmoid', nn.Sigmoid()) # Put outputs between 0 and 1 to stabilise training.

            for i in range(self.gru_layers-1):
                self.fc_latent.append(deepcopy(self.fc_latent[0]))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        sample = mu + (std * eps)
        return sample
        
    def forward(self, x): # Tries to classify & reconstruct input in same genre.
        hidden_states = [torch.zeros((x.shape[0], self.hidden_dim), device=self.device) for i in range(self.gru_layers)] # Initialise hidden state as zeroes
        for i in range(x.shape[1]): # For i in range(len of piece)
            hidden_states[0] = self.encoder[0](self.gru_dropout(x[:, i].squeeze()), hidden_states[0]) # Update hidden state for every timestep
            for j in range(1, self.gru_layers): # Passing hidden states through all GRU layers
                hidden_states[j] = self.encoder[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
        hidden = torch.cat(hidden_states, dim=1)

        if self.bidirectional:
            back_hidden = [torch.zeros((x.shape[0], self.hidden_dim), device=self.device) for i in range(self.gru_layers)]
            for i in range(x.shape[1]-1, -1, -1):
                back_hidden[0] = self.back_encoder[0](self.gru_dropout(x[:, i].squeeze()), back_hidden[0])
                for j in range(1, self.gru_layers):
                    back_hidden[j] = self.back_encoder[j](self.gru_dropout(back_hidden[j-1]), back_hidden[j])
            hidden = torch.cat((hidden, torch.cat(back_hidden, dim=1)), dim=1)

        mu = self.fc_mu(hidden) # Get latent mean from final hidden state
        logvar = self.fc_logvar(hidden) # Get latent logvar from final hidden state
        z = self.reparameterize(mu, logvar) # Reparamaterize & sample from latent space
        genre_pred = self.genre_classifier(z[:, :self.n_genres]) # Applies softmax to first n_genres layers of latent space to get genre predictions
        hidden_states = [self.fc_latent[i](z) for i in range(self.gru_layers)] # Sample from latent space to get initial hidden state for decoder
        
        output_note = x[:, :1, :].squeeze() # Start from first note of input (so not starting from empty note)
        output = x[:, :1, :]
        for i in range(x.shape[1]-1):
            hidden_states[0] = self.decoder[0](self.gru_dropout(output_note), hidden_states[0])
            for j in range(1, self.gru_layers):
                hidden_states[j] = self.decoder[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
            output_note = self.fc_decoder(hidden_states[self.gru_layers-1])
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output, mu, logvar, genre_pred
    
    def vae_train(self, x):
        hidden_states = [torch.zeros((x.shape[0], self.hidden_dim), device=self.device) for i in range(self.gru_layers)] # Initialise hidden state as zeroes
        for i in range(x.shape[1]): # For i in range(len of piece)
            hidden_states[0] = self.encoder[0](self.gru_dropout(x[:, i].squeeze()), hidden_states[0]) # Update hidden state for every timestep
            for j in range(1, self.gru_layers): # Passing hidden states through all GRU layers
                hidden_states[j] = self.encoder[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
        hidden = torch.cat(hidden_states, dim=1)

        if self.bidirectional:
            back_hidden = [torch.zeros((x.shape[0], self.hidden_dim), device=self.device) for i in range(self.gru_layers)]
            for i in range(x.shape[1]-1, -1, -1):
                back_hidden[0] = self.back_encoder[0](self.gru_dropout(x[:, i].squeeze()), back_hidden[0])
                for j in range(1, self.gru_layers):
                    back_hidden[j] = self.back_encoder[j](self.gru_dropout(back_hidden[j-1]), back_hidden[j])
            hidden = torch.cat((hidden, torch.cat(back_hidden, dim=1)), dim=1)

        mu = self.fc_mu(hidden) # Get latent mean from final hidden state
        logvar = self.fc_logvar(hidden) # Get latent logvar from final hidden state
        z = self.reparameterize(mu, logvar) # Reparamaterize & sample from latent space
        genre_pred = self.genre_classifier(z[:, :self.n_genres]) # Applies softmax to first n_genres layers of latent space to get genre predictions
        hidden_states = [self.fc_latent[i](z) for i in range(self.gru_layers)] # Sample from latent space to get initial hidden state for decoder
        
        output = x[:, :1, :]
        for i in range(x.shape[1]-1):
            hidden_states[0] = self.decoder[0](self.gru_dropout(x[:, i, :]), hidden_states[0])
            for j in range(1, self.gru_layers):
                hidden_states[j] = self.decoder[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
            output_note = self.fc_decoder(hidden_states[self.gru_layers-1])
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output, mu, logvar, genre_pred
    
    def sample(self):
        z = torch.randn((128, self.latent_dim), device=self.device)
        hidden_states = [self.fc_latent[i](z) for i in range(self.gru_layers)] # Sample from latent space to get initial hidden state for decoder
        output_note = torch.cat((torch.ones(128, 1), torch.zeros(128, self.input_size-1)), dim=1) # Start from first note of input (so not starting from empty note)
        output = torch.unsqueeze(output_note, 1)
        for i in range(96):
            hidden_states[0] = self.decoder[0](self.gru_dropout(output_note), hidden_states[0])
            for j in range(1, self.gru_layers):
                hidden_states[j] = self.decoder[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
            output_note = self.fc_decoder(hidden_states[self.gru_layers-1])
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output
    
    def generate(self, x, genre_vec): # Generate a piece in style of genre in genre_vec
        z = torch.cat((genre_vec, torch.randn((genre_vec.shape[0], self.latent_dim-self.n_genres), device=self.device)), dim=1)
        hidden_states = [self.fc_latent[i](z) for i in range(self.gru_layers)] # Sample from latent space to get initial hidden state for decoder
        output_note = x[:, :1, :].squeeze() # Start from first note of input (so not starting from empty note)
        output = x[:, :1, :]
        for i in range(x.shape[1]-1):
            hidden_states[0] = self.decoder[0](self.gru_dropout(output_note), hidden_states[0])
            for j in range(1, self.gru_layers):
                hidden_states[j] = self.decoder[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
            output_note = self.fc_decoder(hidden_states[self.gru_layers-1])
            output = torch.cat((output, torch.unsqueeze(output_note, 1)), 1)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, gru_layers, fc_dropout,
                gru_dropout, bidirectional, fc_layers, critic, device):
            super(Discriminator, self).__init__()

            self.fc_dropout = fc_dropout
            self.gru_dropout = nn.Dropout(p=gru_dropout)
            self.device = device
            self.gru_layers = gru_layers
            self.bidirectional = bidirectional
            self.fc_layers = fc_layers
            self.input_size = input_size
            self.hidden_dim = hidden_dim
            self.critic = critic

            if bidirectional: # If bidirectional, hidden dim for producing latent space has to be twice as large due to concatenation of encoder outputs.
                self.discriminator_hidden_dim = 2*gru_layers*hidden_dim
            else:
                self.discriminator_hidden_dim = gru_layers*hidden_dim

            self.discriminator = nn.ModuleList([nn.GRUCell(input_size=self.input_size,
            hidden_size=self.hidden_dim)])
            for i in range(self.gru_layers-1):
                self.discriminator.append(nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))
            
            if bidirectional:
                self.back_discriminator = nn.ModuleList([nn.GRUCell(input_size=self.input_size, 
                hidden_size=self.hidden_dim)])
                for i in range(self.gru_layers-1):
                    self.back_discriminator.append(nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))

            self.fc_discriminator = nn.Sequential()
            for i in range(self.fc_layers-1):
                self.fc_discriminator.add_module('Linear_' + str(i),
                    nn.Linear(in_features=self.discriminator_hidden_dim//2**i,
                    out_features=self.discriminator_hidden_dim//2**(i+1)))
                self.fc_discriminator.add_module('ReLU_' + str(i),
                    nn.LeakyReLU())
                self.fc_discriminator.add_module('Dropout_' + str(i),
                    nn.Dropout(p=self.fc_dropout))

            self.fc_discriminator.add_module('Linear_' + str(self.fc_layers-1),
                nn.Linear(in_features=self.discriminator_hidden_dim//2**(self.fc_layers-1), out_features=1))
            if not self.critic:
                self.fc_discriminator.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        hidden_states = [torch.zeros((x.shape[0], self.hidden_dim), device=self.device) for i in range(self.gru_layers)] # Initialise hidden state as zeroes
        for i in range(x.shape[1]): # For i in range(len of piece)
            hidden_states[0] = self.discriminator[0](self.gru_dropout(x[:, i, :].squeeze()), hidden_states[0]) # Update hidden state for every timestep
            for j in range(1, self.gru_layers):
                hidden_states[j] = self.discriminator[j](self.gru_dropout(hidden_states[j-1]), hidden_states[j])
        hidden = torch.cat(hidden_states, dim=1)
        if self.bidirectional:
            back_hidden = [torch.zeros((x.shape[0], self.hidden_dim), device=self.device) for i in range(self.gru_layers)]
            for i in range(x.shape[1]-1, -1, -1):
                back_hidden[0] = self.back_discriminator[0](self.gru_dropout(x[:, i, :].squeeze()), back_hidden[0])
                for j in range(1, self.gru_layers):
                    back_hidden[j] = self.back_discriminator[j](self.gru_dropout(back_hidden[j-1]), back_hidden[j])
            hidden = torch.cat((hidden, torch.cat(back_hidden, dim=1)), dim=1)
        prediction = self.fc_discriminator(hidden)
        return prediction



class Optimisation:
    def __init__(self, model, discriminator, checkpoint_path, optimiser, disc_optimiser,
            beta, sigma, class_weight, genre_dict, batch_size, vae_mse, generator_loops,
            critic, gp_weight, device, verbose=False):
        self.verbose = verbose
        self.device = device
        self.model = model
        self.discriminator = discriminator
        self.optimiser = optimiser
        self.disc_optimiser = disc_optimiser
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.val_losses = []
        self.vae_mse = vae_mse
        self.generator_loops = generator_loops
        self.mse = nn.MSELoss().to(self.device)
        self.bce = nn.BCELoss().to(self.device)
        self.beta = beta
        self.sigma = sigma
        self.gp_weight = gp_weight
        self.critic = critic
        self.logsigma = math.log(sigma)
        self.class_weight = class_weight
        self.batch_size = batch_size
        self.genre_dict = genre_dict
        self.n_genres = len(genre_dict)
        self.vae_lr = self.optimiser.param_groups[0]['lr']
        self.gan_lr = self.disc_optimiser.param_groups[0]['lr']

    def vae_loss_fn(self, x, x_hat, mu, logvar, genre, genre_pred):
        if self.vae_mse == True:
            loss = self.mse(x, x_hat) # reconstruction loss between input & output, i.e. how similar are they
        else:
            loss = self.bce(x_hat, x)
        class_loss = self.bce(genre_pred, genre)
        kl_div = (0.5 * (mu ** 2 + torch.exp(logvar)/self.sigma - logvar + self.logsigma - 1).sum(axis=1)).mean()
        # KL Div loss, i.e. how similar is prior to posterior distribution. We sum loss over latent variables for each batch & timestep, then mean.
        return loss, self.beta * kl_div, self.class_weight * class_loss
    
    def compute_grad_pen(self, x, x_hat): # Compute gradient penalty for use in WGAN-GP loss function
        eps = torch.rand(x.shape(0), 1, 1).to(self.device)
        eps = eps.expand_as(x)
        interpolation = eps * x + (1-eps) * x_hat

        interp_logits = self.discriminator(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        gradients = autograd.grad(
            outputs = interp_logits,
            inputs = interpolation,
            grad_outputs = grad_outputs,
            create_graph = True,
            retain_graph = True
        )[0]

        gradients = gradients.view(x.shape(0), -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
    
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
        return qn_num.item(), uqn_num.item()

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

    def train_step(self, x, genre_vec, vae, qn=True, upc=False):
        self.optimiser.zero_grad()
        self.model.train()
        self.discriminator.train()
        self.disc_optimiser.zero_grad()
        total_disc_loss = 0
        total_loss = 0
        total_rec_loss = 0
        total_class_loss = 0
        total_kl_loss = 0
        if vae:
            x_hat, mu, logvar, genre_pred = self.model.vae_train(x)
            rec_loss, kl_loss, class_loss = self.vae_loss_fn(x, x_hat, mu, logvar, genre_vec, genre_pred)
            total_rec_loss = rec_loss.item()
            total_class_loss = class_loss.item()
            total_kl_loss = kl_loss.item()
            loss = rec_loss + kl_loss + class_loss
            loss.backward()
            self.optimiser.step()
            total_loss = loss.item()
        else:
            if self.critic:
                x_hat = self.model.generate(x, genre_vec)
                fake_score = self.discriminator(x_hat)
                loss = -fake_score.mean()
                loss.backward()
                self.optimiser.step()
                total_loss = loss.item()

                self.model.eval()

                for _ in range(self.generator_loops):
                    self.disc_optimiser.zero_grad()
                    real_score = self.discriminator(x)
                    fake_score = self.discriminator(x_hat)
                    grad_penalty = self.gp_weight * self.compute_grad_pen(x, x_hat)
                    loss_c = fake_score.mean() - real_score.mean()
                    disc_loss = loss_c + grad_penalty
                    disc_loss.backward()
                    self.disc_optimiser.step()
                    total_disc_loss += disc_loss.item()
                total_disc_loss /= self.generator_loops

            else:
                x_hat = self.model.generate(x, genre_vec)
                fake_pred = self.discriminator(x_hat.detach())
                fake_class = torch.ones_like(fake_pred)
                real_class = torch.zeros_like(fake_pred)
                real_pred = self.discriminator(x)
                disc_loss = self.bce(torch.cat((fake_pred, real_pred)), torch.cat((fake_class, real_class)))
                disc_loss.backward()
                self.disc_optimiser.step()
                total_disc_loss = disc_loss.item()

                for _ in range(self.generator_loops):
                    fake_pred = self.discriminator(x_hat)
                    loss = self.bce(fake_pred, real_class)
                    total_loss += loss.item()
                    loss.backward()
                    self.optimiser.step()
                    self.optimiser.zero_grad()
                    x_hat = self.model.generate(x, genre_vec)

                total_loss /= self.generator_loops

        if qn:
            qn_num, uqn_num = self.measure_qn(x_hat)
            if qn_num+uqn_num != 0:
                qn = qn_num/(qn_num+uqn_num)
            else:
                qn = 0
        return total_loss, total_disc_loss, qn, upc, total_rec_loss, total_class_loss, total_kl_loss
    
    def test_step(self, x, genre_vec):
        self.model.eval()
        x_hat, mu, logvar, genre_pred = self.model(x)
        rec_loss, kl_loss, class_loss = self.vae_loss_fn(x, x_hat, mu, logvar, genre_vec, genre_pred)
        loss = rec_loss + kl_loss + class_loss
        return loss.item(), x_hat, rec_loss.item(), class_loss.item()
    
    def train(self, train_loader, test_loader, writer, n_epochs, eval_every, measure_every, cycle_every, vae_train_proportion):
        genre_vec_dict = {} # Dictionary to hold genre vectors for different genres
        for genre in self.genre_dict:
            genre_vec_dict[genre] = torch.cat((torch.zeros(self.genre_dict[genre]), torch.ones(1), torch.zeros(self.n_genres-(1+self.genre_dict[genre]))))
            # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        running_loss = 0.0
        running_rec_loss = 0.0
        running_class_loss = 0.0
        running_kl_loss = 0.0
        running_disc = 0
        running_qn = 0.0
        tm1 = time.time()
        tm2 = time.time()
        ld = 0
        trn = 0
        i=0
        for epoch in range(1, n_epochs+1):
            if epoch % cycle_every >= cycle_every * vae_train_proportion: # If doing GAN training, need to change optmimiser to GAN lr
                vae = False
                for g in self.optimiser.param_groups:
                        g['lr'] = self.gan_lr
            else:
                vae = True
                for g in self.optimiser.param_groups:
                        g['lr'] = self.vae_lr
            for batch, genre in train_loader:
                if self.verbose:
                    print('Iteration', i)
                tm2 = tm1
                tm1 = time.time()
                ld += tm1 - tm2
                input = batch.to(self.device)
                if self.verbose:
                    print('After batch to device', torch.cuda.memory_allocated(self.device))
                genre_vec = torch.stack([genre_vec_dict[gen] for gen in genre]).to(self.device)
                loss, disc_loss, qn, _, rec_loss, class_loss, kl_loss = self.train_step(input, genre_vec, vae=vae)
                running_loss += loss
                running_rec_loss += rec_loss
                running_class_loss += class_loss
                running_kl_loss += kl_loss
                running_disc += disc_loss
                running_qn += qn
                tm2 = tm1
                tm1 = time.time()
                trn += tm1 - tm2
                i += 1
                if i % measure_every == 0:
                    writer.writerow([i,
                                    running_loss / measure_every,
                                    running_disc / measure_every,
                                    running_qn / measure_every,
                                    running_rec_loss / measure_every,
                                    running_class_loss / measure_every,
                                    running_kl_loss / measure_every,
                                    ld, trn])
                    running_loss = 0.0
                    running_rec_loss = 0.0
                    running_class_loss = 0.0
                    running_kl_loss = 0.0
                    running_disc = 0.0
                    running_qn = 0.0
                    ld = 0
                    trn = 0
            if epoch == 1 or epoch % eval_every == 0:
                print('done epoch {} of {}'.format(epoch, n_epochs))
                eval_losses = []
                eval_classes = []
                eval_recs = []
                first = True
                for batch, genre in test_loader:
                    with torch.no_grad():
                        input = batch.to(self.device)
                        genre_vec = torch.stack([genre_vec_dict[gen] for gen in genre]).to(self.device)
                        loss, out, rec_loss, class_loss = self.test_step(input, genre_vec)
                        if first:
                            out = convert_batch(out, threshold=0.5)
                            proll = convert_pianoroll(out)
                            proll.write('./Outputs/' + str(datetime.datetime.now()) + '.mid')
                            first = False
                        eval_losses.append(loss)
                        eval_classes.append(class_loss)
                        eval_recs.append(rec_loss)
                print('Mean loss for epoch ' + str(epoch) + ': ' + str(mean(eval_losses)))
                print('Reconstruction Loss: ' + str(mean(eval_recs)))
                print('Classification Loss: ' + str(mean(eval_classes)))
                #torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimiser_state_dict': self.optimiser.state_dict(), 'loss': mean(eval_losses)}, self.checkpoint_path + str(datetime.datetime.now()) + '.pt')

    def lr_range_test(self, train_loader, writer, vae, start_lr=-5.0, stop_lr=0.0, lr_step=0.5, measure_every=10, change_every=500):
        genre_vec_dict = {} # Dictionary to hold genre vectors for different genres
        for genre in self.genre_dict:
            genre_vec_dict[genre] = torch.cat((torch.zeros(self.genre_dict[genre]), torch.ones(1), torch.zeros(self.n_genres-(1+self.genre_dict[genre]))))
            # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        lr = 10**start_lr
        for g in self.disc_optimiser.param_groups:
            g['lr'] = lr
        for g in self.optimiser.param_groups:
            g['lr'] = lr
        i = 0
        running_loss = 0.0
        while True:
            for batch, genre in train_loader:
                input = batch.to(self.device)
                genre_vec = torch.stack([genre_vec_dict[gen] for gen in genre]).to(self.device)
                loss, _, _, _ = self.train_step(input, genre_vec, vae)
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
                    for g in self.disc_optimiser.param_groups:
                        g['lr'] = lr