import numpy as np
import pretty_midi as pm


def convert_batch(batch, low_note=19, high_note=97, threshold=0.5): # Converts entire batch to 1 long song, or converts until stop token found.
    note_range = high_note - low_note
    batch = np.array(batch)
    batch = (batch >= threshold).tolist()
    pianoroll = [[] for _ in range(5)]
    for i in range(len(batch)):
        for j in range(len(batch[0])):
            if batch[i][j][0] == 1:
                continue
            if batch[i][j][1] == 1:
                return pianoroll
            for k in range(5):
                pianoroll[k].append([[0]*low_note + batch[i][j][k*note_range+2:(k+1)*note_range+2] + [0]*(128-high_note)])
    return np.asarray(pianoroll)

def convert_dense(batch, instrument_vec_size=19):
    pianoroll = [[] for _ in range(5)]
    for i in range(batch.shape[0]):
        for j in range(batch.shape[1]):
            for k in range(batch.shape[2]):
                if batch[i][j][k] <= 128:
                    pianoroll[k//instrument_vec_size].extend([0]*batch[i][j][k])
                else:
                    pianoroll[k//instrument_vec_size].append(batch[i][j][k]-128)
    return np.asarray(pianoroll)

def convert_pianoroll(pianoroll, tempo=120): # Converts a pianoroll in a list to a prettymidi object to be saved as a MIDI file
    note_time = 60/(tempo*24)
    drums = pm.Instrument(program=0, is_drum=True, name="drums") # Tempo is 120 here by default
    piano = pm.Instrument(program=0, name="piano")
    guitar = pm.Instrument(program=24, name="guitar")
    bass = pm.Instrument(program=32, name="bass")
    strings = pm.Instrument(program=48, name="strings")
    pointer_tensor = np.zeros((5, 128))
    for i in range(pianoroll.shape[1]+1):
        if i < pianoroll.shape[1]:
            current_slice = pianoroll[:, i, :].squeeze()
        else:
            current_slice = np.zeros_like(current_slice)
        pointer_tensor += current_slice
        ending = pointer_tensor * np.logical_not(current_slice)
        for j in range(128):
            if ending[0][j] > 0:
                drums.notes.append(pm.Note(velocity=100, pitch=j, start=(i-ending[0][j])*note_time, end=i*note_time))
            if ending[1][j] > 0:
                piano.notes.append(pm.Note(velocity=100, pitch=j, start=(i-ending[1][j])*note_time, end=i*note_time))
            if ending[2][j] > 0:
                guitar.notes.append(pm.Note(velocity=100, pitch=j, start=(i-ending[2][j])*note_time, end=i*note_time))
            if ending[3][j] > 0:
                bass.notes.append(pm.Note(velocity=100, pitch=j, start=(i-ending[3][j])*note_time, end=i*note_time))
            if ending[4][j] >0:
                strings.notes.append(pm.Note(velocity=100, pitch=j, start=(i-ending[4][j])*note_time, end=i*note_time))
        pointer_tensor *= current_slice
    song = pm.PrettyMIDI(initial_tempo=tempo)
    song.instruments.append(drums)
    song.instruments.append(piano)
    song.instruments.append(guitar)
    song.instruments.append(bass)
    song.instruments.append(strings)
    return song

