#!/usr/bin/env python
# coding: utf-8
# @author Souheib Ben Mabrouk


import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import pandas as pd
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style('darkgrid')
import argparse

from cVAE_architecture2 import cVAE2, loss_vae
from cVAE_train_utils2 import train_cVAE


parser = argparse.ArgumentParser(description='Control the parameters of the model')
parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of the latent space')
parser.add_argument('--beta', type=float, default=1.0, help='Beta value for Beta-VAE')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for the training')
parser.add_argument('--path', type=str, default='./cell_data_balanced_2classes_Wild_PIK3CA/', help='path to the dataset')
parser.add_argument('--input_size', type=int, default=192, help='size in pixels of the input images')
parser.add_argument('--saving_path', type=str, default='./models/', help='path to save the model')

args = parser.parse_args()
beta = args.beta
latent_dim = args.latent_dim
epochs = args.epochs
path = args.path
input_size = args.input_size
saving_path = args.saving_path

print("latent_dim set to ",latent_dim)
print("beta set to ",beta)
print("epochs set to ",epochs)
print("path to dataset set to ",path)
print("input size set to ",input_size)
print("saving_path set to ",saving_path)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 128

# Define a transform to preprocess the data
transform = transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder(root=path, transform=transform)

train_proportion = 0.8
test_proportion = 1 - train_proportion

# Calculate the number of samples for each split
total_samples = len(dataset)
train_samples = int(train_proportion * total_samples)
test_samples = total_samples - train_samples

# Use random_split to create train and test datasets with the calculated proportions
train_dataset, test_dataset = random_split(dataset, [train_samples, test_samples])

# Create DataLoader instances to load batches during training and testing
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X_train = []
Y_train = []
for images, labels in train_loader:
    X_train.append(images)
    Y_train.append(labels)
X_train = torch.cat(X_train, dim=0)
Y_train = torch.cat(Y_train, dim=0)

X_test = []
Y_test = []
for images, labels in test_loader:
    X_test.append(images)
    Y_test.append(labels)
X_test = torch.cat(X_test, dim=0)
Y_test = torch.cat(Y_test, dim=0)

# Print shapes for verification
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


model = cVAE2(num_labels=2, latent_dim=latent_dim,input_size=input_size).to(device)


model = train_cVAE(model, beta=beta, 
                   epochs= epochs, train_loader=train_loader, 
                   test_loader=test_loader, 
                   optimizer=torch.optim.Adamax(model.parameters(), lr=1e-3),
                   criteration = loss_vae, 
                   save_model=True, 
                   saving_path=saving_path,
                   latent_dim=latent_dim)