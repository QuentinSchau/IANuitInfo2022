import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import time
import os 
from torch.utils.data import DataLoader

# class qui implémente le dataset de pytorch
class IstDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        # On a num_features colonnes plus la colonne avec le label associé aux features et l'identifiant
        # On ne prend pas l'identifiant 
        data = np.loadtxt(csv_path,np.float32,delimiter=";", usecols=range(1,num_features+2))
        self.data = data
        self.csv_path = csv_path
        self.transform = transform

    def __getitem__(self, index):
        # le label est à la fin de la ligne dans le fichier
        features = torch.from_numpy(self.data[index][:-1])        
        if self.transform is not None:
            features = self.transform(features)
        label = self.data[index][-1]
        return features, int(label)

    def __len__(self):
        return self.data.shape[0]

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.1
num_epochs = 10
batch_size = 4

# Architecture
num_features = 26
num_hidden_1 = 50
num_hidden_2 = 50
num_hidden_3 = 50
num_hidden_4 = 25
num_classes = 2

def normalize(data):
    std,mean = torch.std_mean(data)
    return (data - mean)/std

class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, num_features, num_classes,num_hidden_1,num_hidden_2,num_hidden_3,num_hidden_4):
        super(MultilayerPerceptron, self).__init__()
        
        ### 1er couche cache
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
            
        ### 2eme couche
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        
        ### 3eme couche
        self.linear_3 = torch.nn.Linear(num_hidden_2, num_hidden_3)
        
        ### 3eme couche
        self.linear_4 = torch.nn.Linear(num_hidden_3, num_hidden_4)
        
        ### Output layer
        self.linear_out = torch.nn.Linear(num_hidden_4, num_classes)
        
    def forward(self, x):
        out = self.linear_1(x)
        out = F.relu(out)
        out = self.linear_2(out)
        out = F.relu(out)
        out = self.linear_3(out)
        out = F.relu(out)
        out = self.linear_4(out)
        out = F.relu(out)
        logits = self.linear_out(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    


def compute_accuracy(net, data_loader):
    net.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits, probas = net(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100
    