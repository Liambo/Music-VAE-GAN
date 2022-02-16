from posixpath import split
import numpy as np
import os
import pypianoroll
import pretty_midi as pm
import random
import torch
import pickle

def construct_dataset(low_note=20, high_note=104): # Does preprocessing and converts songs into numpy arrays & saves as .npz files for dataloader.
    path = os.getcwd() + '/Dataset/Genres/'
    save_path = os.getcwd() + '/Dataset/'
    temp_set = []
    labels = []
    for genre in os.listdir(path):
        if genre.startswith('.'):
            continue
        directory = path + genre
        for file in os.listdir(directory):
            if file.startswith('.'):
                continue
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
                temp_set.append(torch.tensor(songroll[96*i:96*(i+1), :], dtype=torch.float32))
                labels.append(genre)
        print('done', genre)
    temp = list(zip(temp_set, labels))
    random.shuffle(temp)
    temp_set, labels = zip(*temp)
    i=0
    for bar in temp_set:
        torch.save(bar, save_path+'Samples/'+str(i)+'.pt')
        i += 1
    file = open(save_path+'Sample_Labels.p', "wb")
    pickle.dump(labels, file)
    file.close()


construct_dataset()