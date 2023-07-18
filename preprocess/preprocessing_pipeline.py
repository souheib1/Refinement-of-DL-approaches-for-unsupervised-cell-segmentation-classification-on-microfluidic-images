#Preprocessing pipeline
from image_utils import crop,show_images,normalize_image
from preprocessing_utils import vertical_orientation,clahe,eliminate_lines,clip_intensity,remove_background,denoise_image,sigmoid


def preprocess(image,angle=None,crop = False, threshold = None, row_min=0, row_max=-1, col_min=0, col_max=-1, clip_limit=None, clip_value=None, clip_limit_clahe=None, kernel_size=None, nbins=256, weight=0.02, cutoff=.05, gain=1, aff=0, bg=False):
    # apply transformations to the images
    # crop is optional 
    # aff = 1 to plot the different phases of preprocessing
    # bg = True if it is better to remove the background
    
    rotated = vertical_orientation(image,angle=angle)
    clahe_image = clahe(rotated, clip_limit_clahe=clip_limit_clahe, kernel_size=kernel_size, nbins=nbins)
    #lineEliminated = elimate_lines(clahe_image,threshold= (np.max(image) - np.min(image)) / 10000)
    lineEliminated = eliminate_lines(normalize_image(rotated),threshold= threshold)
    if bg: 
        clahe_image = remove_background(clahe_image)
    clip_intensity_image=clip_intensity(lineEliminated,clip_limit=clip_limit,clip_value=clip_value)
    denoised = denoise_image(clip_intensity_image,weight=weight)
    sigmoid_image = denoised
    #sigmoid_image = sigmoid(denoised,cutoff=cutoff, gain=gain)
    if (crop):
        outimage = crop(sigmoid_image,row_min, row_max, col_min, col_max)
    else: 
        outimage = sigmoid_image
    
    if aff == 1: 
        images_list = [image,rotated,clahe_image,clip_intensity_image,denoised,outimage]
        labels = ["image","rotated","clahe","clip_intensity_image","denoised","preprocessed"]

        show_images(images_list,labels)
    
    return(outimage)