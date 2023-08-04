from image_utils import load_images_from_path
from train_utils import train_VAE
import numpy as np



data_path = '../../segmentation/data_for_VAE/220429_ MCF10A  laminAC fibro phallo pattern mars 2022\*.png'
_,crops = load_images_from_path(data_path)
crops = np.expand_dims(crops, axis=1).astype("float32") / 255
batch_size = 32  # You can adjust this based on your hardware and memory limitations
latent_dim = 8
beta = 1
epochs = 50
model_name = "model_zdim_" + str(latent_dim)+"_beta_"+ str(beta)+"_epochs_"+str(epochs)
trained_vae = train_VAE(crops, n_rows=190, n_cols=105, latent_dim=latent_dim, beta=beta, epochs=epochs,
                        batch_size=batch_size, learning_rate=1e-3, validation_split=0.2,
                        plot_history=True, save_model=True, saving_path='./models/'+model_name+".pth")