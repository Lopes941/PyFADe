"""
Functions for creating a fit for ARIMA and ARIMAX models
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
#import pmdarima
import pywt
#from pmdarima import ARIMA

from typing import Literal, Tuple, Union, Any

def znormalize(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """ Returns the z-normalized time series.

        Returns the z-normalized time series or multi-dimensional time series.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Time series to be normalized.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)

        Returns
        -------
        normalized_data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Dataset of same type as input, with normalized dataset.
    """

    # Extracting relevant information depending on the data type
    if isinstance(data,pd.Series):
        normalized_data = (data - data.mean())/data.std(ddof=0)
    elif isinstance(data,pd.DataFrame):
        normalized_data = (data - data.mean())/data.std(ddof=0)
    elif isinstance(data,np.ndarray):
        if data.ndim == 1:
            normalized_data = (data - np.mean(data))/np.std(data)
        else:
            normalized_data = (data - np.mean(data,axis=1))/np.std(data,axis=1)
    elif isinstance(data, list):
        num_dim = len(data)
        normalized_data = [None]*num_dim
        for k in range(num_dim):
            if not isinstance(data, pd.Series):
                raise Exception('Wrong type for data input. Must be dataframe, series, array or list of series!')
            normalized_data[k] = (data[k] - data[k].mean())/data[k].std(ddof=0)
    else:
        raise Exception('Wrong type for data input. Must be dataframe, series, array or list of series!')


    return normalized_data