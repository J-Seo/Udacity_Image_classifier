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

def predict(normal_image, model, topk=5, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_image = torch.from_numpy(normal_image)
        input_image = input_image.unsqueeze(0)
        input_image = input_image.type(torch.FloatTensor)
        input_image = input_image.to(device)
        
        output = model.forward(input_image)
        ps = torch.exp(output)
        
        prob, idx = torch.topk(ps, topk)
        prob = [float(p) for p in prob[0]]
        classify2idx = {v: k for k, v in model.class_to_idx.items()}
        classify = [classify2idx[int(i)] for i in idx[0]]
        
    return prob, classify
    