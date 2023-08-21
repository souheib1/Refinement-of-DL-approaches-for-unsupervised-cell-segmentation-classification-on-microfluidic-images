#!/usr/bin/env python
# coding: utf-8
# @author Souheib Ben Mabrouk

# ## Introduction


import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
sns.set_style('darkgrid')
from clustering_utils import umap,tSNE,compute_most_represented_class_per_cluster,substitute_classes_labels
from VAE_architecture import VAE, loss_vae
from VAE_train_utils import train_VAE


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 128


# ## I. Train the model on MNIST

# Define a transform to preprocess the data
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist_data', train=False, transform=transform, download=True)

# Create data loaders to handle batch processing
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



latent_dim=16
model = VAE(latent_dim=16).to(device)
model = train_VAE(model, beta=1, train_loader=train_loader, test_loader=test_loader, criteration = loss_vae, optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), save_model=True, saving_path='./VAE_models/',latent_dim=16)


latent_dim=32
model_32 = VAE(latent_dim=32).to(device)
model_32 = train_VAE(model_32, beta=1, train_loader=train_loader, test_loader=test_loader, optimizer=torch.optim.Adam(model_32.parameters(), lr=1e-3),criteration = loss_vae, save_model=True, saving_path='./VAE_models/',latent_dim=32)


# ## II. FashionMnist 


# Define a transform to preprocess the data
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST_data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST_data', train=False, transform=transform, download=True)

# Create data loaders to handle batch processing
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


latent_dim = 16
model = VAE(input_size=28,latent_dim=16)
model = train_VAE(model, beta=1, train_loader=train_loader, test_loader=test_loader, optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),criteration = loss_vae, save_model=True, saving_path='./VAEFM_models/')


latent_dim = 32
model = VAE(input_size=28,latent_dim=32)
model = train_VAE(model, beta=1, train_loader=train_loader, test_loader=test_loader, criteration = loss_vae,optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), save_model=True, saving_path='./VAEFM_models/',latent_dim=32)




