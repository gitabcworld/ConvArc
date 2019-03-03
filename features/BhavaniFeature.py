import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np
from features.GlcmFeature import GLCMFeature
from features.LuminanceFeature import LuminanceFeature

class BahavaniFeature(FeatureBase):

    def __init__(self):
        super(BahavaniFeature, self).__init__()
        self.feat1 = LuminanceFeature()
        self.feat2 = GLCMFeature()

    def extract(self, x):
        feats1 = self.feat1.extract(x)
        feats2 = self.feat2.extract(x)
        # concatenate the features
        
        # normalize the histogram
        feats = np.concatenate((feats1,feats2),axis=0)
        feats = feats.astype('float')
        feats /= (feats.sum() + 1e-7)
        return feats