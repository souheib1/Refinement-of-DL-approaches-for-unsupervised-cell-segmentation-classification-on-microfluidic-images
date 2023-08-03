import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from VAE_architecture import BetaVAE


def train_VAE(crops, n_rows=190, n_cols=105, latent_dim=8, beta=1, epochs=20, batch_size=32,
                  learning_rate=1e-3, validation_split=0.2, plot_history=True,
                  save_model=True, saving_path='./newmodel'):


    def __plot_history(history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        model_name = "model_zdim_" + str(latent_dim)+"_beta_"+ str(beta)+"_epochs_"+str(epochs)
        plt.savefig('./models/loss_'+model_name+'.png')
        plt.show()
        

    # Convert the crops to a PyTorch tensor
    crops = torch.tensor(crops)

    # Create an instance of the BetaVAE model
    vae = BetaVAE(beta=beta, latent_dim=latent_dim)

    # Define the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []

    # Split the data into training and validation sets
    split_idx = int((1 - validation_split) * len(crops))
    # Shuffle the data before splitting it
    size_dim_0 = crops.size(0)
    shuffled_indices_dim_0 = torch.randperm(size_dim_0)
    crops = crops[shuffled_indices_dim_0, ...]
    train_data = crops[:split_idx]
    val_data = crops[split_idx:]
    print("the training data has ",train_data.size(dim=0), " patches")
    print("the validation data has ",val_data.size(dim=0), " patches")

    # Training loop
    for epoch in range(epochs):
        vae.train()
        
         # Shuffle the training data
        dim_0 = train_data.size(0)
        shuffled_indices_dim_0 = torch.randperm(dim_0)
        train_data = train_data[shuffled_indices_dim_0, ...]
        
        total_train_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            optimizer.zero_grad()
            reconstructed,z_mean,z_log_var = vae(batch)
            loss, reconstruction_loss, kl_divergence_loss = vae.loss_function(
                reconstructed, batch, z_mean, z_log_var
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_data)
        train_loss_history.append(avg_train_loss)

        # Validation
        vae.eval()
        with torch.no_grad():
            total_val_loss = 0

            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                reconstructed, z_mean, z_log_var = vae(batch)
                loss, _, _ = vae.loss_function(reconstructed, batch, z_mean, z_log_var)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_data)
            val_loss_history.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    if plot_history:
        history = {'loss': train_loss_history, 'val_loss': val_loss_history}
        __plot_history(history)

    if save_model:
        torch.save(vae.state_dict(), saving_path)

    return vae

