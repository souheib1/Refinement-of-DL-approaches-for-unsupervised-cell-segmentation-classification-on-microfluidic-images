# Parts of this code are inspired by the code written by Bettina

from skimage import morphology, filters, exposure
from skimage.restoration import denoise_tv_chambolle
from image_utils import crop,show_images,normalize_image
import numpy as np
import cv2


def clip_intensity(img, clip_limit=None, clip_value=None):
    # Clip image intensities
    
    if clip_limit is None:
        clip_limit = np.quantile(img.flatten(), 0.985)

    if clip_value is None:
        clip_value=clip_limit     

    clipped_img = np.clip(img, 0, clip_limit)
    clipped_img[clipped_img == clip_limit] = clip_value
    return clipped_img


def clahe(img, clip_limit_clahe=None, kernel_size=None, nbins=256):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if clip_limit_clahe is None:
        clip_limit_clahe = .01
    
    if kernel_size is None:
        kernel_size = img.shape[0]//10 
        
    clahe = exposure.equalize_adapthist(img, clip_limit=clip_limit_clahe, kernel_size=kernel_size, nbins=nbins)
    return clahe


def remove_background(image, rb_radius=15, rb_quantile=.99, rb_min_size=100):
    # Remove background using morphological operations
    disk = morphology.disk(rb_radius)
    modified_image = morphology.closing(image, footprint=disk)
    opened_image = morphology.black_tophat(image, footprint=disk)
    thresholded_image = morphology.remove_small_objects(
        opened_image > np.quantile(opened_image.flatten(), rb_quantile),
        min_size=rb_min_size
    )
    modified_copy = modified_image.copy()
    modified_copy[thresholded_image] = image[thresholded_image]
    return modified_copy


def denoise_image(img,weight=0.02):
    # Apply denoising using TV
    denoised_img = denoise_tv_chambolle(img,weight)
    return denoised_img


def find_rotation(img):
    # Find the angle of rotation using Hough line transform based on Canny edge detection

    dst = cv2.Canny(img, (np.max(img) - np.min(img))/10 , np.max(img), None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 80, None, 0, 0)
    if lines is None: 
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 40, None, 0, 0)
    if lines is not None: # consider the line with maximum votes
        max_votes = -np.inf   
        best_line = None
        for line in lines:
            for rho, theta in line:
                if rho > max_votes:
                    max_votes = rho
                    best_line = theta

        angle = best_line * 180 / np.pi 
    else:
        angle = 0
        
    return angle

   
    
def vertical_orientation(img,angle=None):
    if angle is None: 
        angle = find_rotation(img)
        #print("angle of rotation is ",angle)
        
    # Rotate the image using the calculated angle
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle , 1)
    rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=cv2.INTER_NEAREST)
    return rotated
    

def sigmoid(img,cutoff=.5, gain=10):
    # Apply contrast enhancement with sigmoid function
    I_mod = exposure.adjust_sigmoid(img, cutoff=cutoff, gain=gain)
    return I_mod


def eliminate_lines(img, threshold = None):
    if threshold is None: 
        threshold = np.quantile(img.flatten(), 0.4)
    img[img < threshold] = 0
    return img
