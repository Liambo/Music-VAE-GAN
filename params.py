# Constants & data structures - DO NOT EDIT
GENRE_DICT = {'Jazz': 0, 'Country': 1, 'Pop': 2, 'Rock': 3, 'Blues': 4}
N_SAMPLES = 131598

# Parameters - Can edit these
learning_rate = 0.001
weight_decay = 0.000001
kl_beta = 1.0

n_epochs = 500
eval_every=50
train_proportion = 0.9
batch_size = 64
shuffle_train = False
train_num_workers = 4
shuffle_test = False
test_num_workers = 4
