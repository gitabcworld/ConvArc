import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np
from skimage import feature

# Abstract class feature
class LBPFeature(FeatureBase):

    def __init__(self):
        super(LBPFeature, self).__init__()
        self.numPoints = 24
        self.radius = 8

    def extract(self, x):
        # convert to gray
        gray_image = cv2.cvtColor((x*255.).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lbp = feature.local_binary_pattern(gray_image, self.numPoints,self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, self.numPoints + 3),range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)
        return hist
    