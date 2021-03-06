# Constants & data structures - DO NOT EDIT
GENRE_DICT = {'Jazz': 0, 'Country': 1, 'Pop': 2, 'Rock': 3}
N_SAMPLES = 78250
INPUT_SIZE = 402

# Parameters - Can edit these
learning_rate = 10**-2.875
disc_learning_rate = 10**-2.875
disc_weight_decay = 0.000001
weight_decay = 0.000001
kl_beta = 1.0
classifier_weight = 0.03
n_epochs = 50
eval_every = 2
measure_every = 5

train_proportion = 0.9
batch_size = 128
shuffle_train = False
train_num_workers = 2
shuffle_test = False
test_num_workers = 2

cycle_training_every = 50
vae_train_proportion = 0.5
vae_mse = True
generator_loops = 2

hidden_dims = 256
latent_dims = 256
bidirectional = True
fc_layers = 3
gru_layers = 4
fc_dropout = 0.5
gru_dropout = 0.2