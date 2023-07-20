import numpy as np
from skimage.filters import threshold_otsu
from skimage import metrics, exposure, morphology
from scipy import signal, ndimage, stats
import numpy as np
import matplotlib.pyplot as plt

def snr(img):
    #compute SNR based on otsu threhold
    th = threshold_otsu(img)
    m = np.mean(img[img > th])
    sd = np.std(img[img < th])
    return m / sd

def psnr(img, img_mod):
    #compute PSNR
    return metrics.peak_signal_noise_ratio(img, img_mod)

def ssim(img, img_preprocessed):
    #compute SSIM
    return metrics.structural_similarity(img, img_preprocessed)

def norm_grad(img):
    #norm of the gradient (horizontal + vertical)
    return np.linalg.norm(signal.convolve2d(img, [[-1,1]], mode='same') + signal.convolve2d(img, [[-1],[1]], mode='same'))



def cnr(image, h=None, display_maxima=True):
    #Contrast-to-Noise Ratio (CNR) value.
    
    if h is None:
        h = np.mean(image.flatten())

    def calculate_mean_nuclei(image, h, display_maxima):
        peaks = morphology.h_maxima(image, h=h, footprint=None)
        if np.sum(peaks) == 0:
            mean_nuclei = 0
        else:
            mean_nuclei = np.mean(image[peaks])
        if display_maxima:
            print(f"Number of local maxima detected: {np.sum(peaks)}")
            plt.imshow(image, cmap='gray')
            x, y = np.where(peaks.T == True)
            plt.scatter(x, y, marker='*', alpha=.2)
            plt.grid(False)
            plt.show()
        return mean_nuclei
    
    def calculate_mean_background(image,nbins=10000):
        hist, bin_centers = exposure.histogram(image, nbins=nbins)
        mean_bg = bin_centers[np.argmax(hist)]
        return mean_bg
    
    def calculate_sum_variances(image):
        th = np.quantile(image.flatten(), 0.95)
        var_bg = np.var(image[image < th])
        var_fg = np.var(image[image >= th])
        disk = morphology.disk(15)
        image_bth = morphology.black_tophat(image, footprint=disk)
        var_bg = np.var(image_bth)
        var_fg = np.var(image[~np.logical_and(image, image_bth)])
        return np.sqrt(var_bg + var_fg)

    mean_nuclei = calculate_mean_nuclei(image, h, display_maxima)
    mean_background = calculate_mean_background(image)
    sum_variances = calculate_sum_variances(image)
    
    print("Mean nuclei intensity:", mean_nuclei)
    print("Mean background intensity:", mean_background)
    print("Sum of variances:", sum_variances)
    
    cnr_value = np.abs(mean_nuclei - mean_background) / sum_variances
    return cnr_value