"""
Functions for filtering data.
"""

import warnings

import numpy as np
import pandas as pd
import scipy as sp

from scipy.signal import butter
from enum import Enum
from typing import Tuple, Union


class Filter_type(Enum):
    """ Filter selections """
     
    MA = 0, "Moving average filter"
    DIFF = 1, "Finite difference filter"
    BUTTER_LOW = 2, "Low-pass butterworth filter"
    BUTTER_HIGH = 3, "High-pass butterworth filter"
    BUTTER_PASS = 4, "Band-pass butterworth filter"
    

def _extract_values(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Tuple[np.ndarray, int]:
    """ Extracts the numerical values from the data object.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Object from which the data is to be obtained
        subseq_size: int

        Returns
        -------
        values: numpy.ndarray
            Array with the extracted data, with dimensions:
            data.size == (num_dim, series_size)
        num_dim: int
            Number of signals.
    """
    
    if isinstance(data,pd.Series):
        values = data.values
        values = values[np.newaxis,:]
        num_dim = 1
    elif isinstance(data,pd.DataFrame):
        values = data.values.T
        num_dim = values.shape[0]
    else:
        values = data
        if values.ndim == 1:
            values = values[np.newaxis,:]
        num_dim = values.shape[0]


    return values, num_dim

def _filter_ma(lag: int) ->  Tuple[np.ndarray, np.ndarray]:
    """ Calculates the difference equation coeficients for a MA filter.

        Creates a Moving Average filter model, returning the coeficients from the numerator and denominator of its difference equation. Applying this filter to a time series x_i corresponds to:

        y_i = (x_i + x_{i-1} + ... + x_{i-lag}) / (lag+1)

        Parameters
        ----------
        lag: int
            Lag of the moving average filter.

        Returns
        -------
        num_coefs: numpy ndarray
            Array with coefficients from the numerator
        den_coefs: numpy ndarray
            Array with coefficients from the denominator
    """

    num_coefs = np.ones(1, dtype=float)
    den_coefs = np.ones(lag, dtype=float) / (lag+1)

    return num_coefs, den_coefs

def _filter_diff(order: int) ->  Tuple[np.ndarray, np.ndarray]:
    """ Calculates the difference equation coeficients for a backward finite difference filter, without the denominator.

        Creates a backward finite difference filter model, returning the coeficients from the numerator and denominator of its difference equation. Applying this filter to a time series x_i corresponds to:

        order = 1:
            y_i = x_i-x_{i-1}

        order = 2:
            y_i = x_i - 2*x_{i-1} + x_{i-2}

        Parameters
        ----------
        order: int
            Order of the difference.

        Returns
        -------
        num_coefs: numpy ndarray
            Array with coefficients from the numerator
        den_coefs: numpy ndarray
            Array with coefficients from the denominator
    
    """

    from scipy.linalg import pascal

    num_coefs = np.ones(1, dtype=float)
    den_coefs = pascal(order+1, kind='lower')[order, :order+2] * (-1)**np.arange(order+1)
    
    return num_coefs, den_coefs

def _check_keys(required_keys,keys):
    """ Auxiliary function to check if the required inputs for a given filter were provided."""
    for key in required_keys:
        if key not in keys:
            raise ValueError(f'Missing required key: {key}')


def get_filter(filter_type: Filter_type, sampling_freq = 1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the difference equation coefficients of a filter.

        Returns the matrix profile of a time series. The matrix profile is defined by Prof. Eamon Keogh as the minimum distance profile for each subsequence in a time series. It is calculated here via the stumpy library.

        Parameters
        ----------
        filter_type: Filter_type enum
            Type of filter. Different filters require different parameters.

            - MA: Moving Average Filter.

                y_i = (x_i + x_{i-1} + ... + x_{i-lag}) / (lag+1)

                Required parameters: 
                    lag: int
                        Lag of the moving average filter.

            - DIFF: Backward finite difference filter.

                order = 1:
                    y_i = x_i-x_{i-1}
                order = 2:
                    y_i = x_i - 2*x_{i-1} + x_{i-2}

                Required parameters: 
                    order: int
                        Order of the difference, must be positive.

            - BUTTER_LOW: Butterworth low-pass filter.
            
                Required parameters: 
                    order: int
                        Order of the difference, must be positive.
                    cutoff_freq: float
                        Cutoff frequency of filter.

            - BUTTER_HIGH: Butterworth high-pass filter.

                Required parameters: 
                    order: int
                        Order of the difference, must be positive.
                    cutoff_freq: float
                        Cutoff frequency of filter.

            - BUTTER_PASS: Butterworth band-pass filter.

                Required parameters: 
                    order: int
                        Order of the difference, must be positive.
                    cutoff_freq: numpy.ndarray = [float,float]
                        Lower and upper cutoff frequencies of filter.

        sampling_freq: float, optional (default = 1)
            Sampling frequency of signal.

        Returns
        -------
        num_coefs: numpy.ndarray
            Coefficients of the numerator.
        den_coefs: numpy.ndarray
            Coefficients of the denominator.
    """

    match filter_type:

        case Filter_type.MA:
            required_keys = ['lag']
            _check_keys(required_keys,kwargs)
            lag = kwargs['lag']
            num_coefs, den_coefs = _filter_ma(lag)

        case Filter_type.DIFF:

            required_keys = ['order']
            _check_keys(required_keys,kwargs)
            order = kwargs['order']
            num_coefs, den_coefs = _filter_diff(order)

        case Filter_type.BUTTER_LOW | Filter_type.BUTTER_HIGH:
            required_keys = ['order', 'cutoff_freq']
            _check_keys(required_keys,kwargs)
            order = kwargs['order']
            cutoff_freq = kwargs['cutoff_freq']
            if not (isinstance(cutoff_freq,int) or isinstance(cutoff_freq,float)):
                try:
                    cutoff_freq = cutoff_freq[0]
                except:
                    raise AssertionError('Input cutoff_freq must be float!')
                if len(cutoff_freq) > 1:
                    warnings.warn('Input cutoff_freq has more than one element, first one was used!')
            if filter_type ==  Filter_type.BUTTER_LOW:
                num_coefs, den_coefs = butter(order, cutoff_freq,fs=sampling_freq,btype='low')
            else:
                num_coefs, den_coefs = butter(order, cutoff_freq,fs=sampling_freq,btype='high')
            
        case Filter_type.BUTTER_PASS:
            required_keys = ['order', 'cutoff_freq']
            _check_keys(required_keys,kwargs)
            order = kwargs['order']
            cutoff_freq = kwargs['cutoff_freq']


            try:
                if len(cutoff_freq) > 2:
                    warnings.warn('Input cutoff_freq has more than two elements, first two were used!')
                    cutoff_freq = cutoff_freq[:2]
                if not (isinstance(cutoff_freq[0],int) or isinstance(cutoff_freq[0],float)):
                    raise
            except:
                raise AssertionError('Input cutoff_freq must be an iterable of floats!')
            num_coefs,den_coefs = butter(order, cutoff_freq,fs=sampling_freq,btype='pass')
            
        case _:
            raise AssertionError('Filter type not defined!')

    return num_coefs, den_coefs

def apply_filter(data: Union[pd.DataFrame, pd.Series, np.ndarray], num_coefs: np.ndarray, den_coefs: np.ndarray) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Applies a filter created by get_filter() on data.

        Returns the filtered data as the same type that was provided.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Time series whose matrix profile is to be calculated.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)

        num_coefs: numpy.ndarray
            Coefficients of the numerator.

        den_coefs: numpy.ndarray
            Coefficients of the denominator.

        Returns
        -------

        filtered_data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Filtered data. Returned the same type as data.
    """

    filtered_data = data.copy()

    values, num_dim = _extract_values(data)
    
    if num_dim == 1:
        filtered_data = sp.signal.lfilter(num_coefs,den_coefs, values)

    else:
        filtered_data = sp.signal.lfilter(num_coefs,den_coefs, values)

    
    if isinstance(data,pd.DataFrame):
        index = data.index
        filtered_data = pd.DataFrame(filtered_data.T,index=index,columns=[f'Filtered {d}' for d in range(1,num_dim+1)])
    elif isinstance(data, pd.Series):
        index = data.index
        filtered_data = pd.Series(filtered_data,index=index,name='Filtered')

    return filtered_data

