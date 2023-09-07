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

from cVAE_architecture import cVAE, loss_vae
from cVAE_train_utils import train_cVAE

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
batch_size = 128
cmap = plt.cm.get_cmap('viridis', 10)

# Define a transform to preprocess the data
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(
    root='./mnist_data', train=False, transform=transform, download=True)

# Create data loaders to handle batch processing
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

X_train = []
Y_train = []
for images, labels in train_loader:
    X_train.append(images)
    Y_train.append(labels)
X_train = torch.cat(X_train, dim=0).to(device)
Y_train = torch.cat(Y_train, dim=0).to(device)

X_test = []
Y_test = []
for images, labels in test_loader:
    X_test.append(images)
    Y_test.append(labels)
X_test = torch.cat(X_test, dim=0).to(device)
Y_test = torch.cat(Y_test, dim=0).to(device)

# Print shapes for verification
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


model=cVAE(num_labels=10, latent_dim=16,input_size=28).to(device)
model001 = train_cVAE(model, beta=1e-5, 
                      epochs= 20, train_loader=train_loader, 
                      test_loader=test_loader, criteration = loss_vae, 
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                      save_model=True, saving_path='./models_Mnist/',
                      latent_dim=16)


model=cVAE(num_labels=10, latent_dim=16,input_size=28).to(device)
model001 = train_cVAE(model, beta=1e-2, 
                      epochs= 20, train_loader=train_loader, 
                      test_loader=test_loader, criteration = loss_vae, 
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                      save_model=True, saving_path='./models_Mnist/',
                      latent_dim=16)


model=cVAE(num_labels=10, latent_dim=16,input_size=28).to(device)
model001 = train_cVAE(model, beta=1, 
                      epochs= 20, train_loader=train_loader, 
                      test_loader=test_loader, criteration = loss_vae, 
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                      save_model=True, saving_path='./models_Mnist/',
                      latent_dim=16)


model=cVAE(num_labels=10, latent_dim=16,input_size=28).to(device)
model001 = train_cVAE(model, beta=1e2, 
                      epochs= 20, train_loader=train_loader, 
                      test_loader=test_loader, criteration = loss_vae, 
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                      save_model=True, saving_path='./models_Mnist/',
                      latent_dim=16)


model=cVAE(num_labels=10, latent_dim=16,input_size=28).to(device)
model001 = train_cVAE(model, beta=1e5, 
                      epochs= 20, train_loader=train_loader, 
                      test_loader=test_loader, criteration = loss_vae, 
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                      save_model=True, saving_path='./models_Mnist/',
                      latent_dim=16)

model=cVAE(num_labels=10, latent_dim=16,input_size=28).to(device)
model001 = train_cVAE(model, beta=0, 
                      epochs= 20, train_loader=train_loader, 
                      test_loader=test_loader, criteration = loss_vae, 
                      optimizer=torch.optim.Adam(model.parameters(), lr=1e-3), 
                      save_model=True, saving_path='./models_Mnist/',
                      latent_dim=16)


