import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Union, List

from .series import DataFrame
from .core import IDataSetObserver, FeatureGroupInterface, ContinuousStatistics#, MatrixProfile


class FeatureGroups:
    CONST_STATISTICS = 0 
    PY_STATISTICS = 1
    MAT_PROFILE = 2



class WindowBuilder:

    def __init__(self, dataframe: DataFrame):
        self.window = Window()
        self.window.observed_dataframe = dataframe
        dataframe.dataset.add_observer(self.window)

    def set_window_size(self,window_size: int):
        self.window.window_size = window_size

    def set_feature_group(self, added_feature_group_name: FeatureGroups):
        if added_feature_group_name in self.window.dict_of_feature_groups.keys():
            print("Feature already present, not adding again")
            return
        
        for requirement in feature_group_requires[added_feature_group_name]:
            if requirement not in self.window.dict_of_feature_groups.keys():
                self.set_feature_group(requirement)
        
        requirements = self.window.get_requirements(added_feature_group_name)
        self.window.dict_of_feature_groups[added_feature_group_name] = feature_group_callable[added_feature_group_name](self.window.observed_dataframe.dataset, *requirements)

    def build(self):
        if self.window.window_size == 0:
            raise RuntimeError("Must set a window size")
        self.window.update()
        return self.window
        

class Window(IDataSetObserver):

    window_size: int = 0
    observed_dataframe: DataFrame = None
    dict_of_feature_groups: dict = {}

    def __init__(self):
        super().__init__()

    def __dealloc__(self):
        self.observed_dataframe.remove_observer(self)

    def update(self):
        for group_name in self.dict_of_feature_groups.keys():
            self.dict_of_feature_groups[group_name].update(self.observed_dataframe.dataset,self.window_size)

    
    def get_requirements(self, group_name: FeatureGroups):
        data = []
        for requirement in feature_group_requires[group_name]:
            data.append(self.dict_of_feature_groups[requirement])
        return data

    def get_property(self,property: str):
        for feature in self.dict_of_feature_groups.values():
            names = feature.feature_names()
            for name in names:
                if name == property:
                    return feature.get_feature(name)

    @property
    def mean(self):
        return self.get_property('mean')
        
    @property
    def std(self):
        return self.get_property('std')


    
class PythonStatistics(FeatureGroupInterface):

    _name            = 'python_statistics'
    _feature_names   = ['meanpy','stdpy']

    def __init__(self, dataset: DataFrame, cont_mean: ContinuousStatistics):
        super().__init__()
        self.meanpy = np.empty((dataset.data.ndim,0),dtype=float)
        self.stdpy = np.empty((dataset.data.ndim,0),dtype=float)

        self.cont_mean = cont_mean

    def name(self):
        return self._name
    
    def feature_names(self):
        return self._feature_names

    def get_feature(self, name) -> np.ndarray:
        if name == self._feature_names[0]:
            return self.meanpy
        elif name == self._feature_names[1]:
            return self.stdpy
        
    def update(self, dataset: DataFrame, window_size: int):
        window = np.lib.stride_tricks.sliding_window_view(dataset.data, window_size,axis=1)
        # self.meanpy = np.mean(window,axis=2)
        # self.stdpy = np.std(window, axis=2)
        self.meanpy = self.cont_mean.means
        self.stdpy = self.cont_mean.stds

    

feature_group_callable = {
    FeatureGroups.CONST_STATISTICS: ContinuousStatistics,
    FeatureGroups.PY_STATISTICS: PythonStatistics,
    #FeatureGroups.MAT_PROFILE: MatrixProfile,
}

feature_group_requires = {
    FeatureGroups.CONST_STATISTICS: [],
    FeatureGroups.PY_STATISTICS: [FeatureGroups.CONST_STATISTICS],
    FeatureGroups.MAT_PROFILE: [FeatureGroups.CONST_STATISTICS]
}