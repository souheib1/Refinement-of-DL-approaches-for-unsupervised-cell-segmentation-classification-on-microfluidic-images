# Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images
# Research internship 2023 at Telecom Paris



## 1. Adapt image pre-processing to the density of cells imaged

The first step in the image preprocessing pipeline is to rotate the images to ensure that the nuclei are in a fixed orientation. 
To enhance the contrast intensity of the fluorescence within the nuclei, some preprocessing steps are applied :
    
**1. CLAHE** (contrast-limited adaptive histogram equalization)

**2. Normalization** (converting the image to floating-point format, and scaling the pixel values to the range [0, 1] )

**3. Clip pixels intensity range** (adjusts image intensities to enhance contrast by clipping values within a range defined by stats such as percentile to define clip limit)

**4. TV Denoising** (Total Variation denoising is used to reduce noise in images while preserving edges and details. We chose to use Chambolle's method as an iterative algorithm to efficiently compute the denoised image.)


![cc](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/450dfce0-d1ac-428c-a84e-8b62b5aedbcd)

![bb](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/a635f9b0-90db-40fd-ba11-7a830d8d627a)

By leveraging these image pre-processing techniques, we aim to prepare the data for accurate cell segmentation and make it well-suited for Variational Autoencoders. 

## 2. Refine a CNN solution for cell segmentation 
The segmentation is based on preliminary results trained on CellPose [[1]](#1) and error annotation exploring "Human in the loop" training.

![segmentation](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/8c5ac525-6480-410e-aff8-4741fee998df)

The segmentation masks are used to extract cell patches from the image by generating bonding boxes based on cells annotation.

![heatmap_labeling](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/67bfa7c8-9b01-4adf-9882-816dec1164e6)


## 3. Refine VAE (variational auto-encoder) and cVAE (conditional variational auto-encoder) approaches for unsupervised clustering to several datasets with different mutations and experimental conditions.

### 3.1 MNIST dataset: 

![dd](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/16de79c2-d915-42df-b045-b9c5b2ba90c8)

![fff](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/5130039a-d1d0-40b3-9ac3-ef8ee82a0b33)


| Dataset_Z_latentDim  | Accuracy | Silhouette score | Davies-Bouldin Index | Calinski-Harabasz Index |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Mnist_Z_16  | 0.9018  | 0.251985  | 1.8037  | 1107.44  |
| Mnist_Z_32  | 0.9042 |  0.184489 | 2.4797  |  712.183 |


### 3.2 FashionMNIST dataset:

![ee](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/58692ffe-d72c-41b9-8999-cedf2cc319d0)

![ff](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/c8f46a5e-5284-4c8c-9f3c-aed652751686)


| Dataset_Z_latentDim  | Accuracy | Silhouette score | Davies-Bouldin Index | Calinski-Harabasz Index |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| FashioMnist_Z_16  | 0.9  |  0.209225  | 2.02346  | 944.545  |
| FashionMnist_Z_32   |     0.970153      |     0.217652 | 1.75011 |  807.943 |

### 3.3 Wild-type cells vs BRAF mutant:

![tt](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/ec6c051b-8f82-493e-8fe4-710a084ba9fc)

![outputII](https://github.com/souheib1/Refinement-of-DL-approaches-for-unsupervised-cell-segmentation-classification-on-microfluidic-images/assets/73786465/6c798f6b-85cc-4505-9c57-923549a0f590)


| Label | number of cells | % occurrences of wild cells | % occurrences of mutation cells |
| ------------- | ------------- | ------------- | ------------- | 
| 0 | 183 | 53.01% | 46.09% |
| 1 | 345 | 36.52% | 63.48% |
| 2 | 98 | 7.29% | 92.71% |
| 3 | 381 | 72.58% | 27.42% |

## References
<a id="1">[1]</a> 
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). 
Cellpose: a generalist algorithm for cellular segmentation. 
Nature methods, 18(1), 100-106.
