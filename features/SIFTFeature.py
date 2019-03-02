import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
import cv2
import numpy as np

# Abstract class feature
class SIFTFeature(FeatureBase):

    def __init__(self):
        super(SIFTFeature, self).__init__()
        self.max_nfeatures = 10
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.max_nfeatures)

    def extract(self, x):
        # convert to gray
        gray_image = cv2.cvtColor((x*255.).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, des = self.sift.detectAndCompute(gray_image,None)
        if not(des is None):
            # pad zeros if needed
            if des.shape[0]>self.max_nfeatures:
                des = des[0:self.max_nfeatures,:]
            des = np.pad(des,((self.max_nfeatures-des.shape[0],0),(0,0)),mode='constant',constant_values=0)
        else:
            des = np.zeros((self.max_nfeatures,128))
        des = des.flatten()
        return des
