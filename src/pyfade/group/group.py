
from typing import  Self


from .pystatistics import PythonStatistics
from .decomposition import WaveletProfile, WaveletKProfile
from .feature_groups import FeatureGroups, feature_group_requirements, feature_group_callable

from pyfade.window import Window


class FeatureGroupBuilder:
    """
    Builder class for creating a FeatureGroup object.

    This class is used to set a FeatureGroup object to a Window. It automatically checks the requirements, and creates
    the corresponding required FeatureGroups if necessary.

    Parameters
    ----------
    feature_group_name : FeatureGroups
        The name of the feature group that is being added.
    parent_window: Window
        The Window object where this feature will be added.
    replace: bool (optional, default=False)
        Whether to replace the previous feature group of same name. Can be used to refresh properties.


    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyfade 

    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> builder = pyfade.series.DataFrameBuilder(data)
    >>> dataframe = builder.set_interpolation(Interpolation.LINEAR) \
                            .set_extrapolation(Extrapolation.CONSTANT) \
                            .build()

    >>> window_builder = pyfade.window.WindowBuilder(dataframe)
    >>> window = window_builder.set_window_size(2).build()

    >>> group_builder = pyfade.group.FeatureGroupBuilder(pyfade.group.FeatureGroups.ContinuousStatistics,window)
    >>> group_builder.build()

    >>> print(window.mean)
    [[ 1.5.  2.]
    [ 4.5  5.5 ]]
    """

    def __init__(self, feature_group_name, parent_window: Window, replace: bool = False):
        """
        Initializes the FeatureGroupBuilder class, to build a feature by its name and add to the parent_window.

        Creates an IFeatureGroup object and inserts it into the parent_window dictionary of feature groups.

        Parameters
        ----------
        feature_group_name : FeatureGroups
            The name of the feature group that is being added.
        parent_window: Window
            The Window object where this feature will be added.
        replace: bool (optional, default=False)
            Whether to replace the previous feature group of same name. Can be used to refresh properties.

        Raises
        ------
        AttributeError: If the feature_group_name does not correspond to any defined feature groups.
        
        """


        if isinstance(feature_group_name,FeatureGroups):
            if feature_group_name not in FeatureGroups:
                raise AttributeError(f"Feature Group {feature_group_name.value} not found.")
            feature_group_name = feature_group_name.value

        self.requirements = []
        self.parameters = []
        self.again = False
        self.feature_group_name = feature_group_name

        self.parent = parent_window
        if feature_group_name in self.parent.dict_of_feature_groups.keys():
            if not replace:
                print("Feature already present, not adding again")
                self._feature_group = None
                self.again = True
            else:
                self.parent._remove_feature_group(self.parent.dict_of_feature_groups[feature_group_name])
             
    def build_requirements(self):
        """
            Build the required FeatureGroups from this FeatureGroup. For instance, KDP requires MP.
        """

        for requirement in feature_group_requirements[self.feature_group_name]:

            if requirement not in self.parent.dict_of_feature_groups.keys():
                feature_builder = FeatureGroupBuilder(requirement,self.parent)

                for parameter_name, parameter_value in self.parameters[:]:
                    if(parameter_name in feature_group_requirements[requirement]):
                        feature_builder.set_parameter(parameter_name,parameter_value)
                        self.parameters.remove((parameter_name,parameter_value))
                feature_builder.build()
                
            self.requirements.append(self.parent.dict_of_feature_groups[requirement])

    def build(self):
        """
        Build the FeatureGroup into the Window and updates it.

        This method adds the built FeatureGroup into the Window, setting up automatic updates.
        """

        if not self.again:

            self.build_requirements()
            self._feature_group = feature_group_callable[self.feature_group_name](self.parent.observed_dataframe.dataset, self.requirements)
            for parameter_name, parameter_value in self.parameters:
                self._feature_group.set_parameter(parameter_name,parameter_value)

            self.parent._add_feature_group(self._feature_group)
            self._feature_group = None

    def set_parameter(self, parameter_name: str, parameter_value: int | float | bool):
        """
        Sets a parameter of the IFeatureGroup.

        The parameter depends on the IFeatureGroup. Each IFeatureGroup will be responsible for managing
        them and throwing errors if not present.

        Parameters
        ----------
        parameter_name : str
            Parameter name.
        parameter_value: int | float | bool
            Value of the parameter. Its type depends on the parameter being set.

        Returns
        -------
        self : DataFrameBuilder
            For method chaining.
        """

        self.parameters.append((parameter_name,parameter_value))
        
        return self

    def set_parameters(self, new_parameters: dict):

        for parameter_name, parameter_value in new_parameters.items():
            self.set_parameter(parameter_name,parameter_value)

        return self
