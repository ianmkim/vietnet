import os

import matplotlib.pyplot as plt

import torchvision.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from util import download_mnist
from util import test, train, train_many_epochs, train_epochs
from SNN import SpikingNet

batch_size = 1000
DATA_PATH = './data'

training_set, testing_set = download_mnist(DATA_PATH)
train_set_loader = torch.utils.data.DataLoader(
    dataset=training_set,
    batch_size=batch_size,
    shuffle=True)

test_set_loader = torch.utils.data.DataLoader(
    dataset=testing_set,
    batch_size=batch_size,
    shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spiking_model = SpikingNet(device, n_time_steps = 128, begin_eval=0)
#train_many_epochs(spiking_model, device, train_set_loader, test_set_loader)
train_epochs(spiking_model, device, train_set_loader, test_set_loader, 5)
