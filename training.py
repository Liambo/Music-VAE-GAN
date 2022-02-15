import torch
from torch import optim
import os
import datetime
from model import VAE, Optimisation
from file_handling import convert_batch, convert_pianoroll, load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
weight_decay = 0.000001
path = os.getcwd() + '/Dataset/Genres/'
genres = []
for genre in os.listdir(path):
    if not genre.startswith('.'):
        genres.append(genre)
genre_dict = {}
for i in range(len(genres)):
    genre_dict[genres[i]] = i
train_set, test_set = load_dataset(genre_dict)
song = convert_batch(train_set[0][1])
song = convert_pianoroll(song)
save_path = os.getcwd() + '/Outputs/' + train_set[0][0] + '/' + str(datetime.datetime.now()) + '.mid'
song.write(save_path)
input_tensor = torch.from_numpy(train_set[0]).to(device=device)
input_tensor = input_tensor.type(torch.FloatTensor)
vae = VAE(422, 128, 128).to(device)
optimiser = Optimisation(vae, optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay), 1.0, genre_dict, device)
optimiser.train(train_set, test_set, 10)
output_tensor, _, _, _ = vae(input_tensor)
batch = convert_batch(output_tensor)
song = convert_pianoroll(batch)
save_path = os.getcwd() + '/Outputs/' + str(datetime.datetime.now()) + '.mid'
song.write(save_path)