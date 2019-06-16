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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    Im = Image.open(image_path)
    
    width, height = Im.size
    Im = Im.resize((256,256))
    
    width, height = Im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    Im = Im.crop((left, top, right, bottom))
    
    normal_image = np.array(Im) /255
    normal_image = (normal_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    normal_image = normal_image.transpose((2,0,1))
    
    return normal_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax