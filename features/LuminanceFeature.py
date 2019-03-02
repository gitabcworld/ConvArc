import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np
from skimage.color import rgb2yiq

class LuminanceFeature(FeatureBase):

    def __init__(self):
        super(LuminanceFeature, self).__init__()
        self.numPointsHistogram = 256

    def extract(self, x):
        YIQ = rgb2yiq(x) # From RGB to YIQ
        Y = YIQ[:,:,0]
        (hist, _) = np.histogram(Y.ravel(),bins=self.numPointsHistogram)
        # normalize the histogram
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)
        return hist
    