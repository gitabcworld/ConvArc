import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np
from pywt import dwt2, idwt2

# Abstract class feature
class UCIFeature(FeatureBase):

    def __init__(self):
        super(UCIFeature, self).__init__()

    def extract(self, x):
        # convert to gray
        gray_image = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        coeffs = dwt2(gray_image, 'haar')
        cA, (cH, cV, cD) = coeffs
        # variance of Wavelet Transformed image (continuous) 
        # skewness of Wavelet Transformed image (continuous) 
        # curtosis of Wavelet Transformed image (continuous) 
        # entropy of image (continuous)
        fd = None
        return fd
