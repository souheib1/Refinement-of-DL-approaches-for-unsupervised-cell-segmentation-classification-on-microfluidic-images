#!/usr/bin/env python
# coding: utf-8
# @author Souheib Ben Mabrouk


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.utils.data as data_utils 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
batch_size = 128

class cVAE(nn.Module):
    
    def __init__(self, num_labels=10, n_channels=1, convDim1=64, convDim2=128, latent_dim=16, input_size=28, kernel_size=4, beta=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.convDim1 = convDim1
        self.convDim2 = convDim2
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.label = nn.Embedding(num_labels, latent_dim)
        self.beta = beta
        # Calculate the number of output features after encoder convolutions
    
        self.encoder_output_size = self.calculate_encoder_output_size()
        self.decoder_output_size = convDim2 * (self.input_size // 4) * (self.input_size // 4)

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, convDim1, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(convDim1, convDim2, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(convDim2),
            nn.ReLU()
        )

        self.flatten_z_mean = nn.Linear(self.encoder_output_size, out_features=latent_dim)
        self.flatten_z_log_var = nn.Linear(self.encoder_output_size, out_features=latent_dim)

        self.decode_linear = nn.Linear(latent_dim*2, self.encoder_output_size)
        self.decode_2 = nn.ConvTranspose2d(in_channels=convDim2, out_channels=convDim1,
                                           kernel_size=kernel_size, stride=2, padding=1)
        self.decode_1 = nn.ConvTranspose2d(in_channels=convDim1, out_channels=n_channels,
                                           kernel_size=kernel_size, stride=2, padding=1)



    def calculate_encoder_output_size(self):
        # Calculate the output size after encoder convolutions
        def conv_output_size(input_size, kernel_size=self.kernel_size, stride=2, padding=1):
            return (input_size - kernel_size + 2 * padding) // stride + 1
        
        conv1_output_size = conv_output_size(self.input_size, kernel_size=4, stride=2, padding=1)
        conv2_output_size = conv_output_size(conv1_output_size, kernel_size=4, stride=2, padding=1)
        return self.convDim2 * conv2_output_size * conv2_output_size



    def sampling(self,z_mean, z_log_var):
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        # generates random samples from a standard normal distribution
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        #The reparameterization trick
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
        
    def encode(self, x, y):
        y = self.label(y)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean, z_log_var = self.flatten_z_mean(x), self.flatten_z_log_var(x)
        x = self.sampling(z_mean, z_log_var)
        z = torch.cat((x, y), dim = 1)
        return z
    
        
    def decode(self, x, y):
        y = self.label(y)
        z = torch.cat((x, y), dim=1)
        x = self.decode_linear(z)
        x = x.view(x.size(0), self.convDim2, self.input_size // 4, self.input_size // 4)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))
        return reconstruction
    
    
    def forward(self, x, y):
        y = self.label(y)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z_mean, z_log_var = self.flatten_z_mean(x), self.flatten_z_log_var(x)
        x = self.sampling(z_mean, z_log_var)
        z = torch.cat((x, y), dim=1)
        x = self.decode_linear(z)
        x = x.view(x.size(0), self.convDim2, self.input_size // 4, self.input_size // 4)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))          
        return z_mean, z_log_var, reconstruction
    
    
def KL_divergence(z_mean, z_log_var):
    loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / z_mean.size(dim=0)
    return loss


def log_likelihood(x, reconstruction):
    loss = nn.BCELoss(reduction='sum')
    return loss(reconstruction, x)

def mse(x, reconstruction):
    loss = nn.MSELoss(reduction='mean')
    return loss(reconstruction, x)


def loss_vae(beta, original, z_mean, z_log_var, reconstructed):
    total_loss = log_likelihood(original,reconstructed) + beta * KL_divergence(z_mean,z_log_var)
    return total_loss


#def loss_vae2(beta, original, z_mean, z_log_var, reconstructed):
    # img = reconstructed.detach().numpy()
    # if not(np.max(img)<=1.0 or np.min(img) >=0.0 ): 
    #     print("reconstrction issue")
    #     reconstructed = torch.ones(original.shape) - original 
    #     print("Loss = ",log_likelihood(original,reconstructed) + beta * KL_divergence(z_mean,z_log_var))    
    #total_loss = log_likelihood(original,reconstructed) + beta * KL_divergence(z_mean,z_log_var)
    
    #print("MSE = ",mse(original,reconstructed))
    #print("KL = ",KL_divergence(z_mean,z_log_var))
    #return total_loss

