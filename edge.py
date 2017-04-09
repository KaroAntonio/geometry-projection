import cv2
import scipy
from scipy import ndimage
import skimage
import skimage.viewer
import skimage.segmentation
import skimage.data
import skimage.io
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt



def open(filename):

    return cv2.imread(filename)


def edge_detect(im):

    #remove some of the noise in the image
    denoise = ndimage.median_filter(im, 1)

    #run canney to get the edges
    edges = cv2.Canny(denoise,100,400)


    denoise = ndimage.median_filter(edges, 1)
    return edges

def label(im):

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def show(im):

    plt.imshow(im)
    plt.show()

if __name__ == "__main__":

    im = open('cube.jpg')

    edges = edge_detect(im)
    edges = label(edges)
    show(edges)
