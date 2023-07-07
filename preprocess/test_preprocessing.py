# Test functions of preprocessing

from preprocessing_utils import find_rotation
from preprocessing_pipeline import preprocess
from image_utils import load_images_from_path,save_preprocessed_images
import os


# Load images

PATH_DATA = "../Data/"
data_folders = os.listdir(PATH_DATA)

for folder in data_folders: 
    PATH = os.path.join(PATH_DATA, folder, '*.tif')
    image_paths, images = load_images_from_path(PATH)
    #find rotation to align the image vertically
    teta = find_rotation(images[0])
    enhanced_images = [preprocess(image,teta) for image in images]
    experiment = str(PATH.split('/')[2])[:-12]
    print(experiment)
    save_preprocessed_images(enhanced_images, output_path= './' + experiment + '/preprocessed_images/')