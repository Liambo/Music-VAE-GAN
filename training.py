import torch
from torch import optim
import os
import datetime
from model import VAE, Optimisation
from file_handling import convert_batch, convert_pianoroll, load_dataset

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