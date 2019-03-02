import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

class GLCMFeature(FeatureBase):

    def __init__(self):
        super(GLCMFeature, self).__init__()
        self.levels = 256

    def extract(self, x):
        gray_image = cv2.cvtColor((x*255.).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        glcm = greycomatrix(gray_image, [5], [0], 256, symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        energy = greycoprops(glcm, '‘energy’')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        ASM = greycoprops(glcm, 'ASM')[0, 0]
        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)
        return hist


