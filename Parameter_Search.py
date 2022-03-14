import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from model import VAE, Optimisation
from data_loader import Dataset
from params import *
import datetime
import csv

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_idlist = [*range(0, int(N_SAMPLES * train_proportion))]
    test_idlist = [*range(int(N_SAMPLES * train_proportion), N_SAMPLES)]
    train_dataset = Dataset(train_idlist)
    test_dataset = Dataset(test_idlist)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=train_num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=test_num_workers)
    latent_list = [128, 256, 512]
    hidden_list = [128, 256, 512]
    bidirectional_list = [True, False]
    fc_layers_list = [1, 2, 3]
    for latent in latent_list:
        for hidden in hidden_list:
            for bidirectional in bidirectional_list:
                for fc_layers in fc_layers_list:
                    fname = str(latent) + '_' + str(hidden) + '_' + str(bidirectional) + '_' + str(fc_layers) + '_' + str(datetime.datetime.now()) + '.csv'
                    f = open(os.getcwd() + '/experiment_log/' + fname, 'w')
                    writer = csv.writer(f)
                    writer.writerow(['iteration', 'avg_loss', 'avg_qn', 'load_time', 'train_time'])
                    vae = VAE(INPUT_SIZE, hidden, latent, bidirectional, fc_layers, device).to(device)
                    optimiser = Optimisation(vae, os.getcwd() + '/Model_Checkpoints/', optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay),
                                            kl_beta, GENRE_DICT, batch_size, vae_mse, device)
                    optimiser.train(train_loader, test_loader, writer, n_epochs, eval_every, measure_every)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('forkserver')
    main()
