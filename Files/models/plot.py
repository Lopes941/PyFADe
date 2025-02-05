
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from .preprocessing import _extract_values
from typing import Tuple, Union


def _plot_multiple_series(data:pd.Series, height: float, width: float, ylimits: list, xlabel: str, ylabel: str | list,fig,axs) -> Tuple[matplotlib.figure.Figure, list]:

    # Extracting values
    values, num_dim = _extract_values(data)

    # Creating window and axes
    if fig is None or axs is None:
        fig, axs = plt.subplots(num_dim, sharex=True, gridspec_kw={'hspace': 0},figsize=[width,height*num_dim])
        style = 'b-'
    else:
        axs = [ax.twinx() for ax in axs]
        style = 'r-'

    # Generating the ylabel list
    if ylabel is None:
        ylabel = data.name
    if isinstance(ylabel, str):
        if  num_dim == 1:
            ylabel = [ylabel]
    if len(ylabel) != num_dim:
        raise Exception('Size of ylabel must be the same as the number of data given!')

    # Making axs a list, so nothing breaks
    axs = [axs]
    
    # Plotting data
    for k in range(num_dim):
        axs[k].plot(data.index,values[k,:],style)
        if ylimits is not None:
            axs[k].set_ylim(ylimits)
        axs[k].grid(True)
        axs[k].set_ylabel(ylabel[k])
        axs[k].set_xlabel(xlabel)

def _plot_multiple_ndarray(data: np.ndarray, height: float, width: float, ylimits: list, xlabel: str, ylabel: str | list,fig,axs) -> Tuple[matplotlib.figure.Figure, list]:

    # Extracting values
    values, num_dim = _extract_values(data)

    # Creating window and axes
    if fig is None or axs is None:
        fig, axs = plt.subplots(num_dim, sharex=True, gridspec_kw={'hspace': 0},figsize=[width,height*num_dim])
        style = 'b-'
    else:
        axs = [ax.twinx() for ax in axs]
        style = 'r-'

    # Generating the ylabel list
    if ylabel is None:
        ylabel = 'Signal'
    if isinstance(ylabel, str):
        if  num_dim == 1:
            ylabel = [ylabel]
        else:
            text = ylabel
            ylabel = [f'{text} {k}' for k in range(1,num_dim+1)]
    if len(ylabel) != num_dim:
        raise Exception('Size of ylabel must be the same as the number of data given!')
    

    # Making axs a list, so nothing breaks
    if num_dim == 1:
        axs = [axs]

    # Setting the limits of the axes
    xlimits = [0, values.shape[1]]
    
    # Plotting data
    for k in range(num_dim):
        axs[k].plot(values[k,:],style)
        if ylimits is not None:
            axs[k].set_ylim(ylimits)
        axs[k].grid(True)
        axs[k].set_ylabel(ylabel[k])
        axs[k].set_xlim(xlimits)
        axs[k].set_xlabel(xlabel)

    return fig, axs

def _plot_multiple_list(data: list, height: float, width: float, ylimits: list, xlabel: str, ylabel: str | list,fig,axs) -> Tuple[matplotlib.figure.Figure, list]:
    

    # Extracting values
    num_dim = len(data)

    # Creating window and axes
    if fig is None or axs is None:
        fig, axs = plt.subplots(num_dim, sharex=True, gridspec_kw={'hspace': 0},figsize=[width,height*num_dim])
        style = 'b-'
    else:
        axs = [ax.twinx() for ax in axs]
        style = 'r-'

    # Generating the ylabel list
    if ylabel is None:
        ylabel = 'Signal'
    if isinstance(ylabel, str):
        if  num_dim == 1:
            ylabel = [ylabel]
        else:
            text = ylabel
            ylabel = [f'{text} {k}' for k in range(1,num_dim+1)]
    if len(ylabel) != num_dim:
        raise Exception('Size of ylabel must be the same as the number of data given!')
    

    for k, idata in enumerate(data):

        if isinstance(idata, pd.Series):

            axs[k].plot(idata.index,idata.values,style)
            if ylimits is not None:
                axs[k].set_ylim(ylimits)
            axs[k].grid(True)
            axs[k].set_ylabel(ylabel[k])
            axs[k].set_xlabel(xlabel)
    

    return fig, axs

def _plot_multiple_dataframe(data: pd.DataFrame, height: float, width: float, ylimits: list, xlabel: str, ylabel: str | list,fig,axs) -> Tuple[matplotlib.figure.Figure, list]:

    # Extracting values
    values, num_dim = _extract_values(data)

    # Creating window and axes
    if fig is None or axs is None:
        fig, axs = plt.subplots(num_dim, sharex=True, gridspec_kw={'hspace': 0},figsize=[width,height*num_dim])
        style = 'b-'
    else:
        axs = [ax.twinx() for ax in axs]
        style = 'r-'

    # Generating the ylabel list
    if ylabel is None:
        ylabel = data.columns
    elif isinstance(ylabel, str):
        text = ylabel
        ylabel = [f'{text} {k}' for k in range(1,num_dim+1)]
    elif len(ylabel) != num_dim:
        raise Exception('Size of ylabel must be the same as the number of data given!')
    
    # Plotting data
    for k in range(num_dim):
        axs[k].plot(data.index,values[k,:],style)
        if ylimits is not None:
            axs[k].set_ylim(ylimits)
        axs[k].grid(True)
        axs[k].set_ylabel(ylabel[k])
        axs[k].set_xlabel(xlabel)


    return fig, axs


def print_multiple(data: Union[pd.DataFrame, pd.Series, np.ndarray, list], height: float = 1, width: float = 7, ylimits: list = None, xlabel: str = 'Timestamp', ylabel: str | list = None, fig = None, axs = None) -> Tuple[matplotlib.figure.Figure, list]: 
    """ Creates a window with multiple plots alignes vertically.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Set of data to be ploted.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        height: float, optional (default = 1)
            Height of each subplot. Total height of window will be this height times the number of plotted data.
        width: float, optional (default = 7)
            Width of each subplot.
        ylimits: list, optional
            Limits of the y-axis in all plots.
        xlabel: str, optional (default = 'Timestamp')
            Label of the x-axis.
        ylabel: str or list, optional (default = 'Signal')
            Label of each y-axis. If a string is given, will add the number of the plot after the string (i.e. the default value 'Signal' will show as 'Signal 1', 'Signal 2', ...). If a list is given, the labels will be each of the elements.

        Returns
        -------
        fig: matplotlib.figure
            Figure object.
        axs: list[matplotlib.axis]
            List of each of the axis.
    """


    if isinstance(data, pd.DataFrame):
        fig, axs = _plot_multiple_dataframe(data,height,width,ylimits,xlabel,ylabel,fig,axs)
    elif isinstance(data, pd.Series):
        fig, axs = _plot_multiple_series(data,height,width,ylimits,xlabel,ylabel,fig,axs)
    elif isinstance(data, np.ndarray):
        fig, axs = _plot_multiple_ndarray(data,height,width,ylimits,xlabel,ylabel,fig,axs)
    elif isinstance(data, list):
        fig, axs = _plot_multiple_list(data,height,width,ylimits,xlabel,ylabel,fig,axs)
    else:
        return None, None

    return fig, axs