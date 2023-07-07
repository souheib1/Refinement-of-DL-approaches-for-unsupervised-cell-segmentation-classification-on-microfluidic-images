# image_utils.py load images ,crop, normilize and display

import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
from skimage import io


def load_images_from_path(path,channel=2): # channel in {1,2,3,4}
    #Loads and returns the channel of interest from the images from the specified path.
    image_paths = glob(path)
    images = []
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            images.append(image[:,:,channel-1])
                
        except IOError:
            print("Unable to open image:", image_path) 
    
    return image_paths, images


def normalize_image(image):
    # Convert the image to floating-point
    image = image.astype(np.float32)

    # Find the minimum and maximum pixel values
    min_value = np.min(image)
    max_value = np.max(image)

    # Normalize the image
    normalized_image = (image - min_value) / (max_value - min_value)

    return normalized_image


def crop(image, row_min, row_max, col_min, col_max):
    #Crops the input image based on the specified row and column ranges an returns the cropped image.
    return image[row_min:row_max, col_min:col_max]


def show_images(images,labels = None):  
    #Displays a list of images in a row 
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(20, 15))
    for i, image in enumerate(images):
        axes[i].imshow(image, cmap='gray')
        if labels and (len(labels) == len(images)): 
            axes[i].title.set_text(str(labels[i]))
        axes[i].axis('off')
    plt.show()
    

def show_modification(initial, modified, modification = None):
    #Displays the initial and modified images side by side
    fig, axes = plt.subplots(nrows=2, ncols=len(initial), figsize=(17, 8))
    for i in range (len(initial)):
        pl1 = axes[0][i].imshow(initial[i], cmap='gray') 
        pl2 = axes[1][i].imshow(modified[i], cmap='gray')
        axes[0][i].title.set_text('initial image ' + str(i)) 
        if modification != None:
            axes[1][i].title.set_text(str(modification)+ ' image ' + str(i))
        else:
            axes[1][i].title.set_text('modified image ' + str(i))    
        axes[0][i].grid(False)
        axes[1][i].grid(False)
    plt.show()
    
    
def save_preprocessed_images(images, output_path=None):
    #Save List of preprocessed images to the  output path.
    
    if output_path is None:
        output_path = "./" #current folder
    
    for i, image in enumerate(images):
        filename = f"preprocessed_image_{i}.png"
        output_file = os.path.join(output_path, filename)
        image = image * 255
        io.imsave(output_file, image.astype(np.uint8))
        print(f"Preprocessed image {i} saved as {output_file}")