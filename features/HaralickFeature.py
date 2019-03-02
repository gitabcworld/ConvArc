import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np
import mahotas as mt
from skimage import feature

# Abstract class feature
class HaralickFeature(FeatureBase):

    def __init__(self):
        super(HaralickFeature, self).__init__()

    def extract(self, x):
        # convert to gray
        gray_image = cv2.cvtColor((x*255.).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        feats = mt.features.haralick(gray_image)
        return feats
    