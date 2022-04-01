import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from model import VAE, Optimisation
from data_loader import Dataset
from params import *
import datetime
import csv
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = open(os.getcwd() + '/experiment_log/' + str(datetime.datetime.now()) + '.csv', 'w')
    writer = csv.writer(f)

    train_idlist = [*range(0, int(N_SAMPLES * train_proportion))]
    test_idlist = [*range(int(N_SAMPLES * train_proportion), N_SAMPLES)]
    train_dataset = Dataset(train_idlist)
    test_dataset = Dataset(test_idlist)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=train_num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=test_num_workers)

    vae = VAE(INPUT_SIZE, hidden_dims, latent_dims, gru_layers, fc_dropout, gru_dropout, bidirectional, fc_layers, device).to(device)
    optimiser = Optimisation(vae, os.getcwd() + '/Model_Checkpoints/', optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay),
                            kl_beta, classifier_weight, GENRE_DICT, batch_size, vae_mse, device)
    
    print('No. of trainable model parameters: ' + str(sum(p.numel() for p in vae.parameters() if p.requires_grad)))

    optimiser.train(train_loader, test_loader, writer, n_epochs, eval_every, measure_every, cycle_training_every, vae_train_proportion)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
