import pandas as pd
import numpy as np

from .series import DataFrame
from .core import IFeatureGroup

class PythonStatistics(IFeatureGroup):

    @staticmethod
    def static_name():
        return 'python_statistics'
    
    @staticmethod
    def static_feature_names():
        return ['meanpy','stdpy']
    
    @staticmethod
    def static_requirements():
        return ['ContinuousStatistics']
    
    @staticmethod
    def static_parameters():
        return []

    def __init__(self, dataset: DataFrame, requirements: list[IFeatureGroup]):
        super().__init__()
        self.meanpy = np.empty((dataset.data.ndim,0),dtype=float)
        self.stdpy = np.empty((dataset.data.ndim,0),dtype=float)

        self.cont_mean = requirements[0]

    def name(self):
        return PythonStatistics.static_name()
    
    def feature_names(self):
        return PythonStatistics.static_feature_names()
    
    def requirements(self):
        return PythonStatistics.static_requirements()
    
    def parameters(self):
        return PythonStatistics.static_parameters()
    

    def get_feature(self, name) -> np.ndarray:
        features = self.feature_names()
        if name == features[0]:
            return self.meanpy
        elif name == features[1]:
            return self.stdpy
        
    def update(self, dataset: DataFrame, window_size: int):
        # window = np.lib.stride_tricks.sliding_window_view(dataset.data, window_size,axis=1)
        self.meanpy = self.cont_mean.means
        self.stdpy = self.cont_mean.stds
