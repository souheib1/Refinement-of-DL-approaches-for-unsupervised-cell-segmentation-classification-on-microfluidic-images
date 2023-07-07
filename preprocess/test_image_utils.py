# Test functions of utils

from image_utils import load_images_from_path,crop,show_images,show_modification

# Load images
image_paths, images = load_images_from_path("../Data/220429_ MCF10A  laminAC fibro phallo pattern mars 2022 - Tif/*.tif",channel=2)

# Crop images
cropped_images = [crop(image, 100, 300, 200, 400) for image in images]

#Show images

show_modification(images,cropped_images,modification="cropped")

