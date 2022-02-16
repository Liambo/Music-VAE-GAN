from re import I
import numpy as np
import os
import pypianoroll
import pretty_midi as pm


def convert_batch(batch, low_note=20, high_note=104, threshold=0.5): # Converts entire batch to 1 long song, or converts until stop token found.
    note_range = high_note - low_note
    pianoroll = [[] for _ in range(5)]
    batch = batch.tolist()
    for i in range(len(batch)):
        for j in range(len(batch[0])):
            if batch[i][j][0] == 1:
                continue
            if batch[i][j][1] == 1:
                return pianoroll
            for k in range(5):
                noteslist = batch[i][j][k*note_range+2:(k+1)*note_range+2]
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

