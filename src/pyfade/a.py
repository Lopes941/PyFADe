import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Union, List

from .series import DataFrame
from .core import IDataSetObserver, FeatureGroupInterface, ContinuousStatistics, MatrixProfile


class WindowBuilder:

    def __init__(self, dataframe: DataFrame):
        self.window = Window()
        self.window.observed_dataframe = dataframe
        dataframe.dataset.add_observer(self.window)

    def set_window_size(self,window_size: int):
        self.window.window_size = window_size

    def _add_feature_group(self, added_feature_group: FeatureGroupInterface):
        self.window.dict_of_feature_groups[added_feature_group.name()] = added_feature_group

    def build(self):
        if self.window.window_size == 0:
            raise RuntimeError("Must set a window size")
        self.window.update()
        return self.window
        


class Window(IDataSetObserver):

    def __init__(self):
        self.window_size: int = 0
        self.observed_dataframe: DataFrame = None
        self.dict_of_feature_groups: dict = {}
        super().__init__()

    def __del__(self):
        if self.observed_dataframe is not None:
            self.observed_dataframe.remove_observer(self)

    def update(self):
        for group in self.dict_of_feature_groups.values():
            group.update(self.observed_dataframe.dataset,self.window_size)

    
    def get_requirements(self, group_name: str):
        data = []
        for requirement in feature_group_requirements[group_name]:
            data.append(self.dict_of_feature_groups[requirement])
        return data

    def get_feature(self,feature_name: str):
        for group in self.dict_of_feature_groups.values():
            names = group.feature_names()
            for name in names:
                if name == feature_name:
                    return group.get_feature(name)
        raise KeyError(f"Feature '{feature_name}' not found in any feature group.")

    @property
    def mean(self):
        return self.get_feature('mean')
        
    @property
    def std(self):
        return self.get_feature('std')
    
    @property
    def mp(self):
        return self.get_feature('matrix_profile')

    @property
    def index(self):
        return self.observed_dataframe.index[:-self.window_size+1]
    
class FeatureGroupBuilder:

    def __init__(self, feature_group_name: str, parent_window: Window) -> FeatureGroupInterface:

        self.parent = parent_window
        if feature_group_name in self.parent.dict_of_feature_groups.keys():
            print("Feature already present, not adding again")
            self._feature_group = None
            return
        
        for requirement in feature_group_requirements[feature_group_name]:
            if requirement not in self.parent.dict_of_feature_groups.keys():
                feature_builder = FeatureGroupBuilder(requirement,self.parent)
                feature_builder.build()

        requirements = self.parent.get_requirements(feature_group_name)
        self._feature_group = feature_group_callable[feature_group_name](self.parent.observed_dataframe.dataset, requirements)

    def build(self):
        if self._feature_group is not None:
            self.parent.dict_of_feature_groups[self._feature_group.name()] = self._feature_group
            self._feature_group = None

    def set_parameter(self, parameter_name: str, parameter_value):
        self._feature_group.set_parameter(parameter_name,parameter_value)

class PythonStatistics(FeatureGroupInterface):

    @staticmethod
    def static_name():
        return 'python_statistics'
    
    @staticmethod
    def static_feature_names():
        return ['meanpy','stdpy']
    
    @staticmethod
    def static_requirements():
        return []
    
    @staticmethod
    def static_parameters():
        return []

    def __init__(self, dataset: DataFrame, cont_mean: ContinuousStatistics):
        super().__init__()
        self.meanpy = np.empty((dataset.data.ndim,0),dtype=float)
        self.stdpy = np.empty((dataset.data.ndim,0),dtype=float)

        self.cont_mean = cont_mean

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
        window = np.lib.stride_tricks.sliding_window_view(dataset.data, window_size,axis=1)
        # self.meanpy = np.mean(window,axis=2)
        # self.stdpy = np.std(window, axis=2)
        self.meanpy = self.cont_mean.means
        self.stdpy = self.cont_mean.stds

    
FEATURE_GROUP_LIST = [ContinuousStatistics, PythonStatistics, MatrixProfile]

# FeatureNames = []
feature_group_callable = {}
feature_group_requirements = {}
for feature in FEATURE_GROUP_LIST:
    feature_group_callable[feature.static_name()] = feature
    feature_group_requirements[feature.static_name()] = feature.static_requirements()
    # FeatureNames.append(feature.static_name())
