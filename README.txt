Music-VAE-GAN

This project was completed as part of an M.Sci in Computer Science and Mathematics at Durham University. The system makes use of combined VAE and GAN structures to generate original music and transfer MIDI music between genres. System was programmed using Pytorch and trained using a subset of the Lakh Pianoroll Dataset, more information can be found at https://salu133445.github.io/lakh-pianoroll-dataset/

Please install all modules in requirements.txt before running.
The easiest way to use the model is to run interface.py, however please note that
this requires fluidsynth to be installed on your computer.

To train the model, the dataset must be constructed by running prepare_dataset.py first.
