import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

''' This algorithm works best for detection of fluids in MRI of BRAIN TUMORS'''
def entropy_split(histogram):
    total_count = np.sum(histogram)
    # calculating the probability density function
    pdf = histogram / total_count 
    pdf = pdf[pdf != 0] # removing zero values in order not to get errors in the log domain
    # calculating the cumulative sum
    cdf = np.cumsum(pdf)
    
    # compute entropy for black and white parts of the histogram
    hB = np.zeros(histogram.shape)
    hW = np.zeros(histogram.shape)
    
    for t in range(255):
        
        # probabilites of foreground and background
        p0 = cdf[:t]
        p1 = cdf[t:]
        # Black entropy
        hB[t] = -np.sum(p0 * np.log2(cdf[t]))
        
        # White entropy
        pTW = 1 - cdf[t]
        hW[t] = -np.sum(p1 * np.log2(pTW))
    

    entropy_combined = hB + hW
    t_max = np.argmax(entropy_combined)
    
    return t_max

def binarize_maxentropy(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    threshold = entropy_split(hist)
    binary_image = (image > threshold).astype(np.uint8) * 255
    
    return binary_image

# TESTER CODE 
file_path = r"Te-meTr_0000.jpg"
input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

binary_result = binarize_maxentropy(input_image)
plt.figure(figsize=(12, 6))

# Input Image
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Input Image")
plt.axis('off')

# Binarized Image (Max Entropy Threshold)
plt.subplot(1, 2, 2)
plt.imshow(binary_result, cmap='gray')
plt.title("Binarized Image (Max Entropy Threshold)")
plt.axis('off')

plt.tight_layout()
plt.show()
