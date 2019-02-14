from abc import ABC, abstractmethod

# Abstract class feature
class FeatureBase(ABC):

    @abstractmethod
    def extract(self, x):
        pass
    
    @abstractmethod
    def train(self, x, labels):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass

