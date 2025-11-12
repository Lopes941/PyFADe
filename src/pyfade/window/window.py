import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from enum import Enum
from typing import  Self

from ..series.series import DataFrame, plot_data
from ..features import Features
from ..group import FeatureGroups, feature_group_requirements, feature_group_callable
from ..core import IDataSetObserver, IFeatureGroup


HSPACE = 0.2
class Window(IDataSetObserver):
    """
    Window object that holds FeatureGroup data that automatically updates when the dataset changes.

    This class is used to manage features from a multi-dimensional time serie dataset. It is responsible for
    calling automatic updates whenever the dataset changes. It provides properties to access any defined feature.

    It provides read-only easy access to common features, such as rolling mean, rolling standard deviation.

    Each FeatureGroup can only be defined once in the same Window.


    Attributes
    ----------
    window_size: int
        The Window object being built
    observed_dataframe: DataFrame
        The DataFrame object that is being observed by this window.
    dict_of_feature_groups: dict[str, IFeatureGroup]
        Dictionary of the feature groups defined in this window.

    Read-only Properties
    -------------------
    mean: np.ndarray
        Rolling mean.
    stddev: np.ndarray
        Rolling standard deviation.
    mp: np.ndarray
        Matrix Profile of each independent time-series dimension.
    mp_ind: np.ndarray
        Closest index for each window.
    kp: np.ndarray
        K-Dimensional Profile of multi-dimensional time-series.
    kp_ind: np.ndarray
        Sorted indices on K-Dimensional Profile.
    index: np.ndarray
        Index for observed dataset.
        
    Methods
    -------
    build() -> Window:
        Builds the Window object and returns it.
    set_window_size(window_size: int) -> Self:
        Sets the window size for rolling features.

    See Also
    --------
    WindowBuilder: Builder class for creating a Window object.
    DataFrame: The DataFrame object that is being observed.
    FeatureGroupBuilder: Builder class for FeatureGroups within window.
    IFeatureGroup (C++): Interface for FeatureGroups.
    
    """

    def __init__(self):
        """
        Initializes the Window object.
        """

        self.window_size: int = 0
        self.observed_dataframe: DataFrame = None
        self.dict_of_feature_groups: dict[str,IFeatureGroup] = {}
        super().__init__()

    def __del__(self):
        """
        Deallocates the DataFrame object, removing it from the list of observers from dataset.
        """

        if self.observed_dataframe is not None:
            self.observed_dataframe.dataset.remove_observer(self)

    def update(self):
        """
        Runs the update function for every FeatureGroup.

        This method is called every time the dataset is updated. It runs the update functions
        for each FeatureGroup.
        """

        for group in self.dict_of_feature_groups.values():
            group.update(self.observed_dataframe.dataset,self.window_size)

    def _add_feature_group(self, added_feature_group: IFeatureGroup):
        """
        Adds a FeatureGroup object to this window, adding it to the automatic update scheme.

        See Also
        --------
        FeatureGroupBuilder: Builder class for FeatureGroups within window.
        IFeatureGroup (C++): Interface for FeatureGroups.
        """

        added_feature_group.update(self.observed_dataframe.dataset,self.window_size)
        self.dict_of_feature_groups[added_feature_group.name()] = added_feature_group

    def _remove_feature_group(self, removed_feature_group: IFeatureGroup):
        """
        Removes a FeatureGroup object to this window, adding it to the automatic update scheme.

        See Also
        --------
        FeatureGroupBuilder: Builder class for FeatureGroups within window.
        IFeatureGroup (C++): Interface for FeatureGroups.
        """
        del self.dict_of_feature_groups[removed_feature_group.name()]
        del removed_feature_group
        

    def get_feature(self,feature_name: Features | str) -> np.ndarray:
        """
        Gets the requested feature by name.

        Parameters
        ----------
        feature_name: Features
            Name of the requested feature.

        Returns
        -------
        np.ndarray: The requested feature array.

        Raises
        ------
        KeyError: If the feature name is not found in this Window.
        """

        if isinstance(feature_name,Features):
            value = feature_name.value
        else:
            value = feature_name

        for group in self.dict_of_feature_groups.values():
            names = group.feature_names()
            for name in names:
                if name == value:
                    return group.get_feature(name)
                
        raise KeyError(f"Feature '{value}' not found in any feature group.")

    def plot_feature(self,\
            feature_name: str,\
            height: float = 3,\
            width: float = 10,\
            xlimits: list = None,\
            ylimits: list = None,\
            xlabel: str = 'Timestamp',\
            ylabel: str | list = None,\
            title: str = None,\
            fill_na: float = None,\
            fig = None,\
            axs = None,\
            style=None,\
            **kwargs):
        
        labels = self.observed_dataframe.dimensions
        if feature_name in [Features.KProfileInd, Features.KProfileVal, Features.WaveletKP]:
            labels = np.array([f'{k}-KDP' for k in range(1,self.observed_dataframe.dataset.ndim+1)])
        
        data = self.get_feature(feature_name)
        index = self.observed_dataframe.index[:-self.window_size+1]
        

        if fill_na is not None:
            res_data = pd.DataFrame(data.T,index).resample('h').mean().fillna(0)
            data = res_data.values.T
            index = res_data.index
        
        return plot_data(data,\
                        index,\
                        labels,\
                        height,\
                        width,\
                        xlimits,\
                        ylimits,\
                        xlabel,\
                        ylabel,\
                        title,\
                        fig,\
                        axs,\
                        style,\
                        **kwargs)
    
    
    def plot_color(self,\
            feature_name: str,\
            start:int = 0,\
            colormap = 'Reds',\
            scale = 'square',\

            height: float = 3,\
            width: float = 10,\
            xlabel: str = 'Timestamp',\
            ylabel: str | list = None,\
            title: str = None,\
            fig = None,\
            axs = None,\
            style=None,\
            **kwargs):
        
        from matplotlib import pyplot as plt

        ylim = [self.observed_dataframe.data.min(), self.observed_dataframe.data.max()]

        data = self.get_feature(feature_name).T

        num_dim = data.shape[1]

        if scale == 'square':
            data = np.square(data)

        x = self.index.to_numpy()

        data[:start,:] = np.nan

        if fig is None or axs is None:
            fig, axs = plt.subplots(num_dim, sharex=True, gridspec_kw={'hspace': HSPACE},figsize=[width,height*num_dim])
            if style is None:
                style = 'b-'
            if num_dim == 1:
                axs = [axs]

        labels = self.observed_dataframe.dimensions
        if ylabel is not None:
            if isinstance(ylabel, str):
                labels = np.array([ylabel]*num_dim)
            else:
                if len(ylabel) != num_dim:
                    raise AttributeError("Number of terms in ylabel must be equal to the dimension size")
                labels = ylabel

        for k in range(num_dim):

            Z = np.expand_dims(data[:, k], axis=0)

            
            axs[k].pcolormesh(x,ylim,Z[:,1:], cmap=colormap)
            axs[k].grid(False)
            axs[k].tick_params(left=False, labelleft=False)
            axs[k].set_xlabel(xlabel)
        axs[0].set_title(title)

        return fig, axs

    @property
    def mean(self) -> np.ndarray:
        """
        np.ndarray : rolling mean of time-series.
        """

        return self.get_feature(Features.Mean)
        
    @property
    def stddev(self):
        """
        np.ndarray : rolling standard deviation of time-series.
        """
        return self.get_feature(Features.StdDev)
    
    @property
    def mp(self):
        """
        np.ndarray : Matrix Profile values for each time-series.
        """

        return self.get_feature(Features.MatProfileVal)
    
    @property
    def mp_ind(self):
        """
        np.ndarray : Matrix Profile closest index for each time-series.
        """

        return self.get_feature(Features.MatProfileInd)
    
    @property
    def kp(self):
        """
        np.ndarray : K-Dimensional Profile values.
        """

        return self.get_feature(Features.KProfileVal)
    
    @property
    def kp_ind(self):
        """
        np.ndarray : K-Dimensional Profile sorted highest dimension index.
        """

        return self.get_feature(Features.KProfileInd)

    @property
    def index(self):
        """
        np.ndarray : Dataset index.
        """

        return self.observed_dataframe.index[:-self.window_size+1]

class WindowBuilder:
    """
    Builder class for creating a Window object that observes a DataFrame.

    This class is used to set the window size and build the Window object.
    It is designed to be used in a fluent interface style, allowing method chaining.

    Args
    ----
    dataframe (pyfade.DataFrame): The observed dataframe

    Attributes
    ----------
    _window: Window
        The Window object being built

    Methods
    -------
    build() -> Window:
        Builds the Window object and returns it.
    set_window_size(window_size: int) -> Self:
        Sets the window size for rolling features.

    See Also
    --------
    DataFrame: The DataFrame object that is being observed.
    Window: The Window object that is being built.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyfade import DataFrameBuilder, Interpolation, Extrapolation,
    >>>                     WindowBuilder

    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> builder = DataFrameBuilder(data)
    >>> dataframe = builder.set_interpolation(Interpolation.LINEAR) \
                            .set_extrapolation(Extrapolation.CONSTANT) \
                            .build()

    >>> window_builder = WindowBuilder(dataframe)
    >>> window = window_builder.set_window_size(2).build()
    """

    def __init__(self, dataframe: DataFrame):
        """
        Initializes the WindowBuilder object with given data.

        Creates a Window object that observes the given dataframe. This Window object
        has features that are automatically updated when the given dataframe changes.

        Parameters
        ----------
        dataframe : pyfade.DataFrame
            The dataframe to be observed. All features inside the Window object being built
            will automatically updated when this dataframe changes.
        """

        self.window = Window()
        self.window.observed_dataframe = dataframe
        dataframe.dataset.add_observer(self.window)

    def set_window_size(self,window_size: int) -> Self:
        """
        Sets the window size for the Window object.

        The window size used here will be used for the rolling features defined in the window.

        Parameters
        ----------
        window_size : int
            Interpolation method.

        Returns
        -------
        self : WindowBuilder
            For method chaining.

        Raises
        ------
        ValueError: If the window size is not greater than 1.

        """

        if window_size < 1:
            raise ValueError("The set window size must be greater than 1.")

        self.window.window_size = window_size
        return self

    def build(self) -> Window:
        """
        Builds the Window object and returns it.

        This method initializes the Window with the data provided to the WindowBuilder.
        It is expected that the window_size has been set before calling this method.
        
        Returns
        -------
        Window: The built Window object with the dataframe and window_size set.

        Raises
        -------
        RuntimeError: If the window size is not set before calling this method.
        """
         
        if self.window.window_size == 0:
            raise RuntimeError("Must set a window size")
        
        self.window.update()
        return self.window
         
class FeatureGroupBuilder:
    """
    Builder class for creating a FeatureGroup object.

    This class is used to set a FeatureGroup object to a Window. It automatically checks the requirements, and creates
    the corresponding required FeatureGroups if necessary.

    Args
    ----
    feature_group_name (str): The name of the feature group.
    parent_window (Window): The parent Window object.

    Attributes
    ----------
    parent: Window
        The parent Window to which the feature group will be added.
    _feature_group: IFeatureGroup
        The FeatureGroup being created.

    Methods
    -------
    build():
        Builds the FeatureGroup inside the Window object.
    set_parameter(parameter_name: str, parameter_value: int | float | bool) -> Self:
        Sets a parameter from this feature group.

    See Also
    --------
    IFeatureGroup (C++): Interface for FeatureGroups.
    WindowBuilder: Builder class for creating a Window object.
    Window: The Window object where the features are being added.
    

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyfade import DataFrameBuilder, Interpolation, Extrapolation,\
    >>>                    WindowBuilder, FeatureGroupBuilder, FeatureGroups

    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> builder = DataFrameBuilder(data)
    >>> dataframe = builder.set_interpolation(Interpolation.LINEAR) \
                            .set_extrapolation(Extrapolation.CONSTANT) \
                            .build()

    >>> window_builder = WindowBuilder(dataframe)
    >>> window = window_builder.set_window_size(2).build()

    >>> group_builder = FeatureGroupBuilder(FeatureGroups.ContinuousStatistics,window)
    >>> group_builder.build()

    >>> print(window.mean)
    [[ 1.5.  2.]
    [ 4.5  5.5 ]]
    """

    def __init__(self, feature_group_name, parent_window: Window, replace:bool = False):
        """
        Initializes the FeatureGroupBuilder class, to build a feature by its name and add to the parent_window.

        Creates an IFeatureGroup object and inserts it into the parent_window dictionary of feature groups.

        Parameters
        ----------
        feature_group_name : FeatureGroups
            The name of the feature group that is being added.
        parent_window: Window
            The Window object where this feature will be added.

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
        Builds the FeatureGroup into the Window and updates it.

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
