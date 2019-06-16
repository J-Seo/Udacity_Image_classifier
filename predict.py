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
import matplotlib

from preprocessing_data import load_checkpoint, process_image, imshow
from predict_model import predict

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', action='store', type = str, required = True,
                    default = 'flowers/test/3/image_06634.jpg',
                    help='Path to image file ex) "flowers/test/3/image_06634.jpg"')

parser.add_argument('--checkpoint', action='store', type = str, required = True,
                    default = '.',
                    help='Load checkpoint')

parser.add_argument('--top_k', action='store', type = int, default = 5,
                    help='Predict the top K probs')

parser.add_argument('--category_names', action='store', type = str,
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='Path to Json file mapping Categories2names')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for inference, set a switch to true')


args = parser.parse_args()


image_path = args.image_path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
filepath = checkpoint + '/checkpoint.pth'
load_checkpoint(filepath)

normal_image = process_image(image_path)

print("Predicting {} flower names from image".format(top_k))

probs, classes = predict(normal_image, model, top_k, gpu)
classes_name = [cat_to_name[i] for i in classes]

for i in range(len(probs)):
    print("{} : {}".format(classes_name[i], round(probs[i],2))