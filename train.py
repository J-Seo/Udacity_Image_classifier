import argparse
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
import json
from collections import orderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt


from load_dataset import load_data
from load_model import call_model, train_model
# Create Instance
parser = argparse.ArgumentParser()
parser.add_argument('--data_directory',action = 'store',  type = str, required = True, 
                    default = 'flowers',
                    help = 'Traing set, validation set, and test set is saparated')
                    
parser.add_argument('--save_dir', action = 'store', type = str, default = 'save', 
                    help = 'Saving the checkpoint model on directory')
                    
parser.add_argument('--arch', type = str, default = 'vgg16', choice = ['vgg16'],
                    help = 'Choosing the arch what you want')
                    
parser.add_argument('--learning_rate', type = float, default = 0.5, 
                    help = 'Determining the degree of decreasing error on model')
                    
parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'number of hidden nodes')
                    
parser.add_argument('--epochs', type = int, default = 25, 
                    help = 'number of iteration')
                    
parser.add_argument('--gpu', action = 'store_true' , default = False, 
                    help = 'Starting the project on the GPU environmnet')
                    
                    
args = parser.parse_args()

data_dir = args.data_directory
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
epochs = args.epochs
hidden_units = args.hidden_units
gpu = args.gpu

train_set, valid_set, test_set = load_data(data_dir)
model_init = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(model_init, train_set, valid_set, learning_rate, epochs, gpu)

# Make Checkpoint

model.class_to_idx = train_datasets.class_to_idx
checkpoint = {'state_dict' : model.state_dict(),
             'optimizer_state_dict' : optimizer.state_dict(),
              'model' : model,
             'class_to_idx' : model.class_to_idx,
             'criterion': criterion,
             'epochs': epochs} 

torch.save(checkpoint, save_dir + '/checkpoint.pth')

print("Checkpoint saved in {}".format(save_dir))
                    

