from statistics import mean
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import os
import pypianoroll
import datetime
import pretty_midi as pm

# consider bidirectional, start notes have more importance on 2nd pass

def load_dataset(genre, batch_size=64, low_note=20, high_note=104, train_split=0.9):
    directory = os.getcwd() + '/Dataset/Genres/' + genre
    dataset = []
    batch = []
    COUNT = 0 #MAKES LOADING QUICKER FOR TESTING, DELETE WHEN DONE
    try:
        for file in os.listdir(directory):
            COUNT += 1
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
            flags = np.zeros((songroll.shape[1], 2))
            songroll = np.concatenate((flags, songroll[0], songroll[1], songroll[2],songroll[3], songroll[4]), axis = 1)
            start = np.expand_dims(np.zeros(422), 0)
            start[0][0] = 1
            end = np.expand_dims(np.zeros(422), 0)
            end[0][1] = 1
            songroll = np.concatenate((start, songroll, end))
            for i in range(songroll.shape[0]//96): # Split song up into 4 beat segments i.e. 1 bar 
                batch.append(np.array(songroll[96*i:96*(i+1), :]))
                if len(batch) >= batch_size:
                    dataset.append(np.transpose(np.array(batch), (1, 0, 2))) # Transpose so batch is 2nd dimension
                    batch = []
            if COUNT == 10:
                break
    except FileNotFoundError:
        print('Error: no such genre as', genre)
        return
    split_point = int(len(dataset)*train_split)
    print(split_point, 'batches in train set,', len(dataset) - split_point, 'batches in test set.')
    return np.array(dataset[:split_point]), np.array(dataset[split_point:])


def convert_batch(batch, low_note=20, high_note=104, threshold=0.5): # Converts entire batch to 1 long song, or converts until stop token found.
    note_range = high_note - low_note
    pianoroll = [[] for _ in range(5)]
    batch = batch.tolist()
    for i in range(len(batch[0])):
        for j in range(len(batch)):
            if batch[j][i][0] == 1:
                continue
            if batch[j][i][1] == 1:
                return pianoroll
            for k in range(5):
                noteslist = batch[j][i][k*note_range+2:(k+1)*note_range+2]
                for l in range(len(noteslist)):
                    if noteslist[l] >= threshold:
                        noteslist[l] = 1
                    else:
                        noteslist[l] = 0
                pianoroll[k].append([0]*low_note + noteslist + [0]*(128-high_note))
    return pianoroll   

def convert_pianoroll(pianoroll, tempo=120): # Converts a pianoroll in a list to a prettymidi object to be saved as a MIDI file
    note_time = 60/(tempo*24)
    drums = pm.Instrument(program=0, is_drum=True, name="drums") # Tempo is 120 here by default
    piano = pm.Instrument(program=0, name="piano")
    guitar = pm.Instrument(program=24, name="guitar")
    bass = pm.Instrument(program=32, name="bass")
    strings = pm.Instrument(program=48, name="strings")
    for i in range(len(pianoroll[0])):
        for j in range(128):
            if pianoroll[0][i][j] == 1:
                drums.notes.append(pm.Note(velocity=100, pitch=j, start=i*note_time, end=(i+1)*note_time))
            if pianoroll[1][i][j] == 1:
                piano.notes.append(pm.Note(velocity=100, pitch=j, start=i*note_time, end=(i+1)*note_time))
            if pianoroll[2][i][j] == 1:
                guitar.notes.append(pm.Note(velocity=100, pitch=j, start=i*note_time, end=(i+1)*note_time))
            if pianoroll[3][i][j] == 1:
                bass.notes.append(pm.Note(velocity=100, pitch=j, start=i*note_time, end=(i+1)*note_time))
            if pianoroll[4][i][j] == 1:
                strings.notes.append(pm.Note(velocity=100, pitch=j, start=i*note_time, end=(i+1)*note_time))
    song = pm.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(drums)
    song.instruments.append(piano)
    song.instruments.append(guitar)
    song.instruments.append(bass)
    song.instruments.append(strings)
    return song

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
        return output, mu, logvar, z


class Optimisation:
    def __init__(self, model, optimiser, beta, genre_dict):
        self.model = model
        self.optimiser = optimiser
        self.train_losses = []
        self.val_losses = []
        self.mse = nn.MSELoss()
        self.beta = beta
        self.genre_dict = genre_dict
        self.n_genres = len(genre_dict)

    def loss_fn(self, x, x_hat, mu, logvar):
        print(x.shape)
        mse_loss = self.mse(x[:,:,:-self.n_genres], x_hat) # reconstruction loss between input & output, i.e. how similar are they
        kl_div = (0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1).sum(axis=1)).mean()
        # KL Div loss, i.e. how similar is prior to posterior distribution. We sum loss over latent variables for each batch & timestep, then mean.
        
        return mse_loss + self.beta * kl_div
    
    def train_step(self, x, latent_genre_vec, vae=True):
        self.optimiser.zero_grad()
        self.model.train()
        if vae:
            x_hat, mu, logvar, z = self.model(x, latent_genre_vec)
            loss = self.loss_fn(x, x_hat, mu, logvar)
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
    def test_step(self, x, latent_genre_vec):
        x_hat, mu, logvar, z = self.model(x, latent_genre_vec)
        loss = self.loss_fn(x, x_hat, mu, logvar)
        return loss.item()
    
    def train(self, train_set, test_set, device, n_epochs=50, eval_every=5, genre='Jazz'):
        genre_num = self.genre_dict[genre]
        genre_vec = np.concatenate((np.zeros((train_set[0].shape[0], train_set[0].shape[1], genre_num)), np.ones((train_set[0].shape[0], train_set[0].shape[1], 1)), np.zeros((train_set[0].shape[0], train_set[0].shape[1], self.n_genres-(1+genre_num)))), axis=2)
        # Constructs genre vector of 0's with a 1 at position of genre from "genre_dict". This allows conditional encoding for CVAE by concatenating genre info onto end of inputs. We create this vector once at the start, rather than for every batch.
        latent_genre_vec = torch.concat((torch.zeros((train_set[0].shape[1], genre_num)), torch.ones((train_set[0].shape[1], 1)), torch.zeros((train_set[0].shape[1], self.n_genres-(1+genre_num)))), axis=1).to(device=device)
        # Constructs genre vector for latent space now, rather than creating a new one for every batch.
        for epoch in range(1, n_epochs+1):
            for batch in train_set:
                input = torch.from_numpy(np.concatenate((batch, genre_vec), axis=2).astype('float32')).to(device=device)
                # Constructs input by concatenating genre vector to end of every input vector & converting entire matrix to float32 for compatability.
                self.train_step(input, latent_genre_vec)
                print('done batch')
            print('done epoch {} of {}'.format(epoch, n_epochs))
            if epoch % eval_every == 0:
                eval_losses = []
                for batch in test_set:
                    input = torch.from_numpy(batch).to(device=device)
                    loss = self.test_step(input, latent_genre_vec)
                    eval_losses.append(loss)
                print('losses for epoch', epoch)
                print(eval_losses)
                print('mean loss: ' + mean(eval_losses))


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
path = os.getcwd()
genre = "jazz"
genre_dict = {"Blues": 0, "Country": 1, "Jazz": 2, "Rock": 3, "Pop": 4}
learning_rate = 0.001
weight_decay = 0.000001
train_set, test_set = load_dataset('Jazz', 64, 20, 104)
song = convert_batch(train_set[0])
song = convert_pianoroll(song)
save_path = path + '/' + genre + '/' + str(datetime.datetime.now()) + '.mid'
song.write(save_path)
input_tensor = torch.from_numpy(train_set[0]).to(device=device)
input_tensor = input_tensor.type(torch.FloatTensor)
vae = VAE(422, 128, 128)
optimiser = Optimisation(vae, optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay), 1.0, genre_dict)
optimiser.train(train_set, test_set, device, 10)
output_tensor, _, _, _ = vae(input_tensor)
batch = convert_batch(output_tensor)
song = convert_pianoroll(batch)
save_path = path + '/' + genre + '/' + str(datetime.datetime.now()) + '.mid'
song.write(save_path)