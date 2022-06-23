import torch
from params import *
from model import VAE
from file_handling import convert_batch, convert_pianoroll
import os
from midi2audio import FluidSynth
import datetime
import time
import vlc

def generate_song(vae):
    genre_dict = {'0' : 'Jazz', '1':'Country', '2':'Pop', '3':'Rock'}
    while True:
        print('In which genre would you like to generate your song? \n0: Jazz \n1: Country \n2: Pop \n3: Rock')
        opt = input('Response: ')
        if opt in ['0', '1', '2', '3']:
            print('Generating... Please wait!')
            song = vae.sample(int(opt), 4).detach()
            song = convert_batch(song)
            song = convert_pianoroll(song)
            date = str(datetime.datetime.now())
            song.write('./Outputs/' + date + '.mid')
            fs = FluidSynth()
            fs.midi_to_audio('./Outputs/'+date+'.mid', './Outputs/'+date+'.wav')
            print('Saved at /Outputs/'+date+'.wav')
            path = os.getcwd()
            p = vlc.MediaPlayer('file:///'+path+'/Outputs/'+date+'.wav')
            p.play()
            time.sleep(10)
            p.stop()
            break
        else:
            print('Please pick a valid choice.')

def transfer_song(vae, device):
    genre_dict = {'0' : 'Jazz', '1':'Country', '2':'Pop', '3':'Rock'}
    while True:
        print('Which genre would you like to transfer from? \n0: Jazz \n1: Country \n2: Pop \n3: Rock')
        opt = input('Response: ')
        if opt in ['0', '1', '2', '3']:
            transfer_song2(vae, device, 0, genre_dict[opt])
            break
        else:
            print('Please pick a valid choice.')

def classify_song(vae, device):
    genre_dict = {'0' : 'Jazz', '1':'Country', '2':'Pop', '3':'Rock'}
    while True:
        print('Which genre of song would you like to classify? \n0: Jazz \n1: Country \n2: Pop\n3: Rock')
        opt = input('Response: ')
        if opt in ['0', '1', '2', '3']:
            classify_song2(vae, device, 0, genre_dict[opt])
            break
        else:
            print('Please pick a valid choice.')

def transfer_song2(vae, device, i, genre):
    filelist = os.listdir('./Dataset/Transfer_Songs/'+genre)
    while True:
        print('Please choose a song to transfer:')
        index = 0
        for j in range(i, i+5):
            print(str(index) + ': ' + filelist[j].split('<SEP>')[0] + ' - ' + filelist[j].split('<SEP>')[1])
            index += 1
        print('5: Next\n6: Prev')
        opt = input('Response: ')
        if opt in ['0', '1', '2', '3', '4']:
            while True:
                print('Which genre would you like to transfer to? \n0: Jazz \n1: Country \n2: Pop \n3: Rock')
                opt2 = input('Response: ')
                if opt2 not in ['0', '1', '2', '3']:
                    print('Please pick a valid choice.')
                else:
                    print('Transferring... Please wait!')
                    inp = torch.load('./Dataset/Transfer_Songs/' + genre + '/' + filelist[index+int(opt)]).to(device)
                    song = vae.transfer(inp.unsqueeze(0), GENRE_DICT[genre], int(opt)).detach()
                    print(song.shape)
                    song = convert_batch(song)
                    print(song.shape)
                    song = convert_pianoroll(song)
                    date = str(datetime.datetime.now())
                    song.write('./Outputs/' + date + '.mid')
                    fs = FluidSynth()
                    fs.midi_to_audio('./Outputs/'+date+'.mid', './Outputs/'+date+'.wav')
                    print('Saved at /Outputs/'+date+'.wav')
                    path = os.getcwd()
                    p = vlc.MediaPlayer('file:///'+path+'/Outputs/'+date+'.wav')
                    p.play()
                    time.sleep(10)
                    p.stop()
                    break
            break
        elif opt == '5':
            transfer_song2(vae, device, i+5, genre)
            break
        elif opt == '6':
            transfer_song2(vae, device, i-5, genre)
            break
        else:
            print('Please pick a valid choice.')
    
def classify_song2(vae, device, i, genre):
    genre_dict = {0 : 'Jazz', 1:'Country', 2:'Pop', 3:'Rock'}
    filelist = os.listdir('./Dataset/Transfer_Songs/'+genre)
    while True:
        print('Please choose a song to classify:')
        index = 0
        for j in range(i, i+5):
            print(str(index) + ': ' + filelist[j].split('<SEP>')[0] + ' - ' + filelist[j].split('<SEP>')[1])
            index += 1
        print('5: Next\n6: Prev')
        opt = input('Response: ')
        if opt in ['0', '1', '2', '3', '4']:
            print('Classifying... Please wait!')
            inp = torch.load('./Dataset/Transfer_Songs/' + genre + '/' + filelist[index+int(opt)]).to(device)
            pred = vae.classify(inp.unsqueeze(0))
            predgenre = genre_dict[torch.argmax(pred).item()]
            predprob = torch.max(pred).item()
            print('I think this song is in the ' + predgenre + ' genre with ' + str(100 * predprob)[:5] + '% confidence!')
            break
        elif opt == '5':
            classify_song2(vae, device, i+5, genre)
            break
        elif opt == '6':
            classify_song2(vae, device, i-5, genre)
            break
        else:
            print('Please pick a valid choice.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(INPUT_SIZE, hidden_dims, latent_dims, gru_layers, fc_dropout, gru_dropout, bidirectional, fc_layers, device).to(device)
vae.load_state_dict(torch.load('./Model_Checkpoints/True_1.0_2022-04-27 00_18_36.431460.pt', map_location=device)['model_state_dict'])
vae.eval()

while True:
    print('Would you like to: \n0: Generate Song \n1: Transfer Song \n2: Classify Song \n3: Exit')
    opt = input('Response: ')
    if opt == '0':
        generate_song(vae)
    elif opt == '1':
        transfer_song(vae, device)
    elif opt == '2':
        classify_song(vae, device)
    elif opt == '3':
        break
    else:
        print('Please pick a valid choice.')
