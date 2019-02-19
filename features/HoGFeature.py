import os
from features.baseFeature import FeatureBase
from sklearn import svm
from joblib import dump, load
from skimage.feature import hog

# Abstract class feature
class HoGFeature(FeatureBase):

    def __init__(self):
        super(HoGFeature, self).__init__()
        #self.clf = svm.LinearSVC()
        self.clf = svm.SVC(probability=True, gamma='scale')

    def extract(self, x):
        fd, hog_image = hog(x, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L2-Hys')
        return fd
    
    def train(self, x, labels):
        self.clf.fit(x, labels)

    def predict(self, x):
        return self.clf.predict_proba(x)

    def load(self, path):
        dump(self.clf, os.path.join(path,'clf.joblib'))

    def save(self, path):
        self.clf = load(os.path.join(path,'clf.joblib')) 
