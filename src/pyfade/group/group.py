
from enum import Enum

from ..core import ContinuousStatistics, MatrixProfile, KProfile

from .pystatistics import PythonStatistics
from .decomposition import WaveletProfile, WaveletKProfile


"""
List of all feature groups that can be added to Window.

ANY NEW FEATUREGROUPS DEFINED BY THE USER MUST BE ADDED HERE, AND ONLY HERE!
"""
FEATURE_GROUP_LIST = [ContinuousStatistics,
                      PythonStatistics,
                      MatrixProfile,
                      KProfile,
                      WaveletProfile,
                      WaveletKProfile]


"""
FeatureGroups: Enum with feature groups names.
feature_group_callable: Dictionary with callable classes.
feature_group_requirements: Dictionary with the requirements from each feature group
"""
FeatureGroups = Enum(
    'FeatureGroups',
    {
        cls.__name__: cls.static_name()
        for cls in FEATURE_GROUP_LIST
    })

feature_group_callable = {}
feature_group_requirements = {}
for feature in FEATURE_GROUP_LIST:
    feature_group_callable[feature.static_name()] = feature
    feature_group_requirements[feature.static_name()] = feature.static_requirements()