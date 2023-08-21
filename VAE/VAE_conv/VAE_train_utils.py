#!/usr/bin/env python
# coding: utf-8
# @author Souheib Ben Mabrouk



import numpy as np 
import torch
import matplotlib.pyplot as plt
from VAE_architecture import loss_vae,VAE

model = VAE() 
device ='cuda:0' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, beta, criterion, optimizer, data_loader):
    
    train_loss_per_epoch = []
    model.train()
    for x_batch,_ in data_loader:
        x_batch = x_batch.to(device)    
        z_mean, z_log_var, reconstruction = model(x_batch)
        loss = criterion(beta=beta, original=x_batch.to(device).float(), z_mean=z_mean, z_log_var=z_log_var, reconstructed=reconstruction)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_per_epoch.append(loss.item())

    return np.mean(train_loss_per_epoch), z_mean, z_log_var, reconstruction

def eval_epoch(model, beta, criterion, optimizer, data_loader):
    
    val_loss_per_epoch = []
    model.eval()
    with torch.no_grad():
        for x_val, _ in data_loader:
            x_val = x_val.to(device)
            z_mean, z_log_var, reconstruction = model(x_val)
            loss = criterion(beta=beta, original=x_val.to(device).float(), z_mean=z_mean, z_log_var=z_log_var, reconstructed=reconstruction)
            val_loss_per_epoch.append(loss.item())
    return np.mean(val_loss_per_epoch), z_mean, z_log_var, reconstruction


def train_VAE(model, train_loader, test_loader, beta, criteration = loss_vae, learning_rate=1e-3, optimizer = torch.optim.Adam(model.parameters(), lr=1e-3), epochs=15, batch_size=128,  plot_history=True, latent_dim=16,
                  save_model=True, saving_path='./VAE_models/'):


    def __plot_history(history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        model_name = "VAE_MNIST_zdim_" + str(latent_dim)+"_epochs_"+str(epochs)
        plt.savefig(saving_path+ 'loss_'+model_name+'.png')
        plt.show()

    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []

    # Training loop
    for epoch in range(epochs):
        
        # Train
        train_loss, z_mean, z_log_var, reconstruction = train_epoch(model, beta=beta, criterion=criteration, optimizer=optimizer, data_loader=train_loader)
        train_loss_history.append(train_loss/batch_size)                                            
        # Validation
        eval_loss, z_mean, z_log_var, reconstruction = eval_epoch(model, beta=beta, criterion=criteration, optimizer=optimizer, data_loader=test_loader)
        val_loss_history.append(eval_loss/batch_size) 
        
        eval_loss/=batch_size
        train_loss/=batch_size
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss :.4f} - Val Loss: {eval_loss:.4f}")

    if plot_history:
        history = {'loss': train_loss_history, 'val_loss': val_loss_history}
        __plot_history(history)

    if save_model:
        model_name = "VAE_MNIST_zdim_" + str(latent_dim)+"_epochs_"+str(epochs)
        torch.save(model.state_dict(), saving_path+model_name+".pth")

    return model