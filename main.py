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

from pacs_dataset import PACS

from PIL import Image
from tqdm import tqdm

import numpy as np

from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt

# Clone github repository with data
# if not os.path.isdir('./Homework3-PACS'):
#   !git clone https://github.com/MachineLearning2020/Homework3-PACS.git

transformation = transforms.Compose([ transforms.Resize(230),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalizes tensor with mean and standard deviation of IMAGENET
])
PACS_PATH = 'Homework3-PACS/PACS/'
pacs = PACS(root=PACS_PATH, transform=transformation, num_workers=4) # add batch size


