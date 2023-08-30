# Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images
# Research internship 2023 at Telecom Paris



## 1. Adapt image pre-processing to the density of cells imaged

The first step in the image preprocessing pipeline is to rotate the images to ensure that the nuclei are in a fixed orientation. This step helps avoid orientation learning issues when using Cellpose and Variational autoencoders later on. Additionally, the RGB images are transformed into grayscale images, focusing on utilizing the lamin fluorescence, corresponding to the second image channel. 

To enhance the contrast intensity of the fluorescence within the nuclei, some preprocessing steps are applied :
    
**1. CLAHE** (contrast-limited adaptive histogram equalization)

**2. Normalization** (converting the image to floating-point format, and scaling the pixel values to the range [0, 1] )

**3. Clip pixels intensity range** (adjusts image intensities to enhance contrast by clipping values within a range defined by stats such as percentile to define clip limit)

**4. TV Denoising** (Total Variation denoising is used to reduce noise in images while preserving edges and details. We chose to use Chambolle's method as an iterative algorithm to efficiently compute the denoised image.)


![cc](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/450dfce0-d1ac-428c-a84e-8b62b5aedbcd)

![bb](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/a635f9b0-90db-40fd-ba11-7a830d8d627a)

By leveraging these image pre-processing techniques, we aim to prepare the data for accurate cell segmentation and make it well-suited for Variational Autoencoders. 

## 2. Refine a CNN solution for cell segmentation 
based on preliminary results trained on CellPose [[1]](#1) and error annotation exploring "Human in the loop" training.

3. Develop an "outliers" filter to detect erroneous cell patches (i.e. patches with "out of range" intensities, "multi-cell" patches).

4. Refine a preliminary VAE (variational auto-encoder) approach pre-trained on several cell types for unsupervised clustering to several datasets with different experimental conditions.


## References
<a id="1">[1]</a> 
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). 
Cellpose: a generalist algorithm for cellular segmentation. 
Nature methods, 18(1), 100-106.
