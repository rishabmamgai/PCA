import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import floor, sqrt
from PIL import Image
import numpy as np


def load_data(file):
    features = loadmat(file)

    return features['X']


def plot_faces(X):
    """
        Prepares a pixel matrix from the feature vector of an example, 
        constructs the image from the pixel matrix and displayes it on
        a subplot.

        Note - These images are saved in the current directory

        1. Using order = 'F' for reshaping the featue vector as it is distributed in that order
        2. mode='L' specifies the type of image which is 3x8 pixels black and white
    """

    m, n = X.shape
    width = floor(sqrt(n))
    height = floor(n / width)
    
    fig = plt.figure(figsize=(5, 5))

    for i in range(0, 25):
        # Shifting the negative values to right
        X[i, :] = X[i, :] + abs(min(X[i, :]))

        # Reshaping the example to get pixel matrix
        image_mat = np.reshape(X[i, :], (height, width), order='F')

        # Contructing image from pixel matrix
        image = Image.fromarray(np.int8(image_mat), mode='L')
        #image.save(f"Image-{i+1}.jpg")

        # Loading the image
        #img = mpimg.imread(f"Image-{i+1}.jpg")
        
        fig.add_subplot(5, 5, i+1)
        imgplot = plt.imshow(image, cmap='gray')

    plt.show()
