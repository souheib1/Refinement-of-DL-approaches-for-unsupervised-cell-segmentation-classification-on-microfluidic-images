import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np


class Sampling(nn.Module):
    """
    Implementation of the reparameterization trick (to make the sampling process differentiable) used in VAEs to sample from the
    posterior distribution of the latent space variables (z) given the inputs (z_mean, z_log_var).

    Parameters:
        inputs (tuple): A tuple containing two tensors : z_mean and z_log_var.

    Returns:
        torch.Tensor: A tensor containing the reparameterized latent variables z.
                      The shape of the output tensor is the same as z_mean and z_log_var.
    """

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        # generates random samples from a standard normal distribution 
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        #The reparameterization trick
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
    
    
    
class Encoder(nn.Module):
    """
    Encoder neural network that maps input image crops to a triplet (z_mean, z_log_var, z).
    """

    def __init__(self, n_rows=190 ,n_cols=105 ,n_channels=1, h_dim3 = 32, h_dim2 = 64, h_dim1 = 128, latent_dim=32):
        super(Encoder, self).__init__()
        
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(n_channels * n_rows * n_cols, h_dim1)
        self.fc1 = nn.Linear(h_dim1, h_dim2)
        self.fc2 = nn.Linear(h_dim2, h_dim3)
        
        self.dense_mean = nn.Linear(h_dim3, self.latent_dim)
        self.dense_log_var = nn.Linear(h_dim3, self.latent_dim)
        self.sampling = Sampling()

    def forward(self, inputs):
        x = self.flatten(inputs)
        x = F.relu(self.dense(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    


class Decoder(nn.Module):
    """
    Decoder neural network that maps latent variables (z) back to the original image space.
    """

    def __init__(self, n_rows=190 ,n_cols=105 ,n_channels=1, h_dim3 = 32, h_dim2 = 64, h_dim1 = 128, latent_dim=32):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = n_rows * n_cols
  
        

        # Decoder layers
        self.dense = nn.Linear(self.latent_dim, h_dim3)
        self.fc1 = nn.Linear(h_dim3, h_dim2)
        self.fc2 = nn.Linear(h_dim2, h_dim1)
        self.fc3 = nn.Linear(h_dim1, n_channels * n_rows * n_cols)
        

    def forward(self, z):
        x = F.relu(self.dense(z))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x.view(-1, self.n_channels, self.n_rows, self.n_cols)



class BetaVAE(nn.Module):
    
    def __init__(self, beta = 1.0, n_rows=190 ,n_cols=105 ,n_channels=1, h_dim3 = 32, h_dim2 = 64, h_dim1 = 128, latent_dim=32):
        super(BetaVAE, self).__init__()

        self.encoder = Encoder(n_rows, n_cols, n_channels, h_dim3, h_dim2, h_dim1, latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(n_rows, n_cols, n_channels, h_dim3, h_dim2, h_dim1, latent_dim)
        self.beta = beta

    def forward(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed_image = self.decoder(z)
        return reconstructed_image,z_mean,z_log_var

    def loss_function(self, reconstructed, original, z_mean, z_log_var):

        # Reconstruction loss (pixel-wise mean squared error)
        reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum')

        # KL-divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss

        return total_loss, reconstruction_loss, kl_divergence_loss


