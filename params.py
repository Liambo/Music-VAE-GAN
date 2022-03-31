# Constants & data structures - DO NOT EDIT
GENRE_DICT = {'Jazz': 0, 'Country': 1, 'Pop': 2, 'Rock': 3, 'Blues': 4}
N_SAMPLES = 32411
INPUT_SIZE = 392

# Parameters - Can edit these
learning_rate = 10**-2.875
weight_decay = 0.000001
kl_beta = 1.0
n_epochs = 50
eval_every = 2
measure_every = 5

train_proportion = 0.9
batch_size = 512
shuffle_train = False
train_num_workers = 4
shuffle_test = False
test_num_workers = 4

cycle_training_every = 50
vae_train_proportion = 0.5
vae_mse = True

hidden_dims = 256
latent_dims = 256
bidirectional = True
fc_layers = 2
gru_layers = 3