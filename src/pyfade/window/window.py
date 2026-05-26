import pandas as pd
import numpy as np

from typing import  Self

from ..series.series import DataFrame, plot_data
from ..features import Features
from ..core import IDataSetObserver, IFeatureGroup


HSPACE = 0.2
class Window(IDataSetObserver):
    """
    Window object that holds FeatureGroup data that automatically updates when the dataset changes.

    This class is used to manage features from a multi-dimensional time serie dataset. It is responsible for
    calling automatic updates whenever the dataset changes. It provides properties to access any defined feature.

    It provides readonly easy access to common features, such as rolling mean, rolling standard deviation.

    Each FeatureGroup can only be defined once in the same Window.

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

        This method is called AUTOMATICALLY every time the dataset is updated. It runs the update functions
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
        Get the requested feature by name.

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
        """
        Plot a given feature by its name.

        Parameters
        ----------
        feature_name: Features
            Name of the requested feature.

        Returns
        --------
        fig: matplotlib.Figure object
            Figure object.
        axs: list[matplotlib.Axes object]
            List of axes objects.

        Raises
        ------
        KeyError: If the feature name is not found in this Window.
        """
        
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

    Parameters
    ----------
    dataframe : pyfade.DataFrame
            The dataframe to be observed. All features inside the Window object being built
            will automatically updated when this dataframe changes.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyfade.series import DataFrameBuilder, Interpolation, Extrapolation
    >>> from pyfade.window import WindowBuilder

    >>> data = np.array([[1, 2, np.nan], [4, np.nan, 6]])
    >>> builder = DataFrameBuilder(data)
    >>> dataframe = builder.set_interpolation(Interpolation.LINEAR).set_extrapolation(Extrapolation.CONSTANT).build()

    >>> window_builder = WindowBuilder(dataframe)
    >>> window = window_builder.set_window_size(2).build()
    """

    def __init__(self, dataframe: DataFrame):
        """
        Initialize the WindowBuilder object with given data.

        Create a Window object that observes the given dataframe. This Window object
        has features that are automatically updated when the given dataframe changes.

        """

        self.window = Window()
        self.window.observed_dataframe = dataframe
        dataframe.dataset.add_observer(self.window)

    def set_window_size(self,window_size: int) -> Self:
        """
        Set the window size for the Window object.

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
        Build the Window object and returns it.

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
         