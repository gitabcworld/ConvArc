from abc import ABC, abstractmethod

# Abstract class feature
class FeatureBase(ABC):

    @abstractmethod
    def extract(self, x):
        pass


