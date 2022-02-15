import numpy as np
import os
import pypianoroll
import pretty_midi as pm

def load_dataset(genre_dict, batch_size=64, low_note=20, high_note=104, train_split=0.9):
    path = os.getcwd() + '/Dataset/Genres/'
    train_set = []
    test_set = []
    for genre in genre_dict:
        temp_set = []
        batch = []
        directory = path + genre
        for file in os.listdir(directory):
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
                    temp_set.append((genre, np.transpose(np.array(batch), (1, 0, 2)))) # Transpose so batch is 2nd dimension
                    batch = []
        split_point = int(len(temp_set)*train_split)
        train_set += temp_set[:split_point]
        test_set += temp_set[split_point:]
        print('finished loading {} genre: {} train batches, {} test batches'.format(genre, len(temp_set[:split_point]), len(temp_set[split_point:])))
    print(len(train_set), 'batches in train set,', len(test_set), 'batches in test set.')
    return train_set, test_set


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

