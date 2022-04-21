import torch
import os
import pickle

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, id_list):
        'Initialization'
        self.id_list = id_list
        self.path = os.getcwd() + '/Dataset'
        self.labels = pickle.load(open(self.path+'/Sample_Labels.p', 'rb'))
        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.id_list)

  def __getitem__(self, index):
        'Generates one sample of data'
        id = self.id_list[index]
        x = torch.load(self.path + '/Samples/' + str(id) + '.pt')
        y = self.labels[id]
        return x, y

class DenseDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, min_id, max_id, device):
        'Initialization'
        self.data = torch.load('./Dataset/full.pt')[min_id:max_id].to(device)
        self.labels = pickle.load(open(self.path+'/Sample_Labels.p', 'rb'))
        

  def __len__(self):
        'Denotes the total number of samples'
        return self.data.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.data[index]
        y = self.labels[index]
        return x, y