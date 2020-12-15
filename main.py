import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from torch.autograd import Function
from torchvision.models.utils import load_state_dict_from_url

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

from pacs_dataset import PACS, print_photo
from model import DANN

from PIL import Image
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt


PACS_PATH = 'Homework3-PACS/PACS/'
SOURCE_DOMAIN = 'photo'
TARGET_DOMAIN = 'art'
VAL_DOMAINS = ['cartoon', 'sketch']
BATCH_SIZE = 128


# Clone github repository with data
# if not os.path.isdir('./Homework3-PACS'):
#   !git clone https://github.com/MachineLearning2020/Homework3-PACS.git


def fit_simple(model, epochs, train_dataloader, val_dataloader=None, optimizer=None, loss_function=None, scheduler=None):
    model.train()

    for epoch in range(epochs):
        for batch in train_dataloader:
            print_photo(batch)
            return
        pass

        eval(model, val_dataloader)
    pass

def fit_with_adaptation():
    pass

def eval(model, val_dataloader):
    model.eval()
    pass


if __name__ == '__main__':

    transformation = transforms.Compose([ transforms.Resize(230),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalizes tensor with mean and standard deviation of IMAGENET
    ])


    pacs = PACS(root=PACS_PATH, transform=transformation, num_workers=4, batch_size=BATCH_SIZE)
    dann = DANN(pretrained=True, num_domains=2, num_classes=7)

    fit_simple(dann, epochs=5, train_dataloader=pacs[SOURCE_DOMAIN], val_dataloader=pacs[TARGET_DOMAIN])

