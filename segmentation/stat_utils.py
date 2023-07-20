import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from skimage import measure
import numpy as np
from matplotlib.patches import Rectangle



def bound_box(image, labeled_mask, aff=True, edgecolor='red', linewidth=1):
    """
    Plot bounding boxes around regions in the image corresponding to the labeled mask.

    Args:
        image (array-like): Input image as a 2D array or image-like object.
        labeled_mask (array-like): Labeled mask of the regions in the image as a 2D array of integers.
        aff (bool, optional): Flag to enable or disable plotting. If True, bounding boxes will be plotted. Default is True (enabled).
        edgecolor (str, optional): Color of the bounding box edges in the plot. Default is 'red'.
        linewidth (float, optional): Width of the bounding box edges in the plot. Default is 1.

    Returns:
        list: List of bounding boxes, where each bounding box is represented as [label, (min_row, min_col, max_row, max_col)].
    """
    props = measure.regionprops(labeled_mask)
    bboxs = []

    for prop in props:
        label = prop.label
        min_row, min_col, max_row, max_col = prop.bbox
        bboxs.append([label, (min_row, min_col, max_row, max_col)])
        
    if aff:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image, cmap='gray')
        ax.grid(False)
        ax.axis('off')

        for bbox in bboxs:
            label, (min_row, min_col, max_row, max_col) = bbox
            rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                             fill=False, edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)
            # ax.text(min_col, min_row, str(label), color='white', fontsize=8,
            #         verticalalignment='top', bbox={'color': 'red', 'pad': 0, 'alpha': 0.5})

    plt.title("Bounding boxes of the cells")
    plt.show()
    return bboxs



def histogram_distbox_Feature(data_list, feature):
    """
    A function to plot the distribution and box plot of a feature
    
    Parameters:
        data_list (list): A list containing the data for the specified feature.
        feature (str): The name of the feature for labeling the plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    ax1 = sns.histplot(data=data_list, ax=axes[0], kde=True, bins=30)
    ax2 = sns.boxplot(data_list, ax=axes[1],whis=2.5)
    ax1.set_xlabel(feature)
    ax1.set_title("Histogram and KDE")
    ax2.set_xlabel(feature)
    ax2.set_title("Box Plot")
    plt.show()
    print("mean " + feature ,np.mean(data_list))
    
    
    
def properties_ROI(img, mask,aff=True):
    """
    Calculate the area, height, and width of each bounding box of the regions of interest (ROI),
    and optionally generate and display histograms of these properties.

    Parameters:
        img (numpy.ndarray): The input image.
            An array representing the image to analyze.

        mask (numpy.ndarray): Binary mask indicating regions of interest (ROI).
            The mask should have the same dimensions as the input image (img).

        aff (bool, optional): Whether to display histograms of the calculated properties.

    Returns:
        tuple: A tuple containing three lists, each representing the calculated properties of
        the bounding boxes of ROIs.
    """
    props = measure.regionprops(mask)
    areas = []
    heights = []
    widths = []
    
    for prop in props:    
        min_row, min_col, max_row, max_col = prop.bbox
        areas.append(prop.area)
        heights.append(max_row - min_row)
        widths.append(max_col - min_col)
    
    if aff:
        histogram_distbox_Feature(areas, "area")
        histogram_distbox_Feature(heights, "height")
        histogram_distbox_Feature(widths, "width")
    
    return(areas,heights,widths)



def plot_features(data,feature="Value"):
    """
    Create a scatter plot to visualize values of images corresponding to different experiences.

    Parameters:
        data (dict): A dictionary where the keys are tuples representing the experience and image,
                     and the values are the extracted values.

    Returns:
        None: This function displays the plot using `matplotlib.pyplot.show()`.
    """
    
    experiences = []
    images = []
    values = []

    for (experience, image), value in data.items():
        experiences.append(experience)
        images.append(image)
        values.append(value)
    
    # Create a color map based on unique experiences
    unique_experiences = list(set(experiences))
    num_unique_experiences = len(unique_experiences)
    color_map = plt.get_cmap('jet', num_unique_experiences)

    # Convert experiences to numeric labels for color mapping
    experience_labels = [unique_experiences.index(exp) for exp in experiences]
    plt.scatter(images, values, c=experience_labels, cmap=color_map)

    # Add colorbar and set labels
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(num_unique_experiences))
    cbar.set_ticklabels(unique_experiences)
    plt.xlabel('Image')
    plt.ylabel(feature)
    plt.title('Values of Images by Experience')
    plt.show()
