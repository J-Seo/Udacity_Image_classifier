#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from workspace_utils import active_session


def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
   
    # Define training, validation, testing datasets.
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) 

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle =True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_datasets, batch_size = 64)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64)
    
    print("Loading Data from {} is complete!".format(data_dir))
    return train_dataloaders, validation_dataloaders, validation_dataloaders

