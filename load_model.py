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

def call_model(arch, hidden_units):
    
    if arch.lower() == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        model = models.densenet121(pretrained = True)
        

    for param in model.parameters():
        param.requires_grad = False
    
    if arch.lower() == 'vgg13':
        classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                   nn.Dropout(0.1),
                                   nn.ReLU(),
                                   nn.Linear(hidden_units, 102),
                                   nn.Dropout(0.1),
                                   nn.LogSoftmax(dim = 1))
    
    else: 
        classifier = nn.Sequential(nn.Linear(1024, 256),
                      nn.ReLU(),
                      nn.Dropout(0.1),
                      nn.Linear(256,102),
                      nn.LogSoftmax(dim=1))
    
    
    model.classifier = classifier
    print("Build the model on {} arch and {} hidden units".format(arch, hidden_units))
    
    return model

# Validation 

def valid_testing(model, data_set, criterion, device):
    loss = 0 
    accuracy = 0 
    with torch.no_grad():
        for inputs, labels in iter(data_set):
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            
            loss += batch_loss.item()
            ps = torch.exp(logps)
            equals = (labels.data == ps.max(dim = 1)[1])
            accuracy += equals.type(torch.FloatTensor).mean()
            
    return loss, accuracy

# Training model

def train_model(model, train_set, valid_set, learning_rate, epochs, gpu):
    
    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # GPU Mode
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    
    # init_parameters
    print_every = 10
    steps = 0
    running_loss = 0
    train_accuracy = 0
    
    print("It's time to Learn!")
    with active_session():
        for epoch in range(epochs):
        
        model.train()
    
        print(epoch)
        for inputs, labels in iter(train_set):
            
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            ps = torch.exp(logps)
            equals = (labels.data == ps.max(dim = 1)[1])
            train_accuracy += equals.type(torch.FloatTensor).mean()
            
            if steps % print_every == 0:
                
                model.eval()
                
                with torch.no_grad():
                    valid_loss, valid_accuracy = valid_testing(valid_set)
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Train Accuracy: {train_accuracy/print_every:.3f}.."
                  f"Valid loss: {valid_loss/len(vaild_set):.3f}.."
                  f"Valid accuracy: {valid_accuracy/len(valid_set):.3f}..")
                  
                
                running_loss = 0
                train_accuracy = 0
                model.train()
        print("Training is completed")
    return model, optimizer, criterion

                      
                      
                


