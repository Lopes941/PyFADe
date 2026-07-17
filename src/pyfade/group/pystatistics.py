
import numpy as np

from .feature_groups import IFeatureGroup,  ContinuousStatistics


from pyfade.features import Features


class PythonStatistics(IFeatureGroup):

    @staticmethod
    def static_name():
        return 'python_statistics'
    
    @staticmethod
    def static_feature_names():
        return [Features.Meanpy.value,
                Features.Stdpy.value]
    
    @staticmethod
    def static_requirements():
        return [ContinuousStatistics.static_name()]
    
    @staticmethod
    def static_parameters():
        return []

    def __init__(self, dataset, requirements: list[IFeatureGroup]):
        super().__init__()

        from ..features.pystatistics import Meanpy, Stdpy

        self.meanpy: Meanpy = Meanpy()
        self.stdpy: Stdpy = Stdpy()

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
            return self.meanpy.feature
        elif name == features[1]:
            return self.stdpy.feature
        
    def update(self, dataset, window_size: int):
        self.meanpy.feature = self.cont_mean.means
        self.stdpy.feature = self.cont_mean.stds
