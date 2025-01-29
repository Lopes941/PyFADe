import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import butter
import matplotlib
import matplotlib.pyplot as plt
import datetime
import stumpy
from typing import Literal, Tuple
import warnings
from enum import Enum



class Filter_type(Enum):
    """ Filter selections """
     
    MA = 0, "Moving average filter"
    DIFF = 1, "Finite difference filter"
    BUTTER_LOW = 2, "Low-pass butterworth filter"
    BUTTER_HIGH = 3, "High-pass butterworth filter"
    BUTTER_PASS = 4, "Band-pass butterworth filter"
    

def _extract_values(data: pd.DataFrame | pd.Series | np.ndarray) -> Tuple[np.ndarray, int]:
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

    def check_keys(required_keys,keys):
        """ Auxiliary function to check if the required inputs for a given filter were provided."""
        for key in required_keys:
            if key not in keys:
                raise ValueError(f'Missing required key: {key}')



    match filter_type:

        case Filter_type.MA:
            required_keys = ['lag']
            check_keys(required_keys,kwargs)
            lag = kwargs['lag']
            num_coefs, den_coefs = _filter_ma(lag)

        case Filter_type.DIFF:

            required_keys = ['order']
            check_keys(required_keys,kwargs)
            order = kwargs['order']
            num_coefs, den_coefs = _filter_diff(order)

        case Filter_type.BUTTER_LOW | Filter_type.BUTTER_HIGH:
            required_keys = ['order', 'cutoff_freq']
            check_keys(required_keys,kwargs)
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
            check_keys(required_keys,kwargs)
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

def apply_filter(data: pd.DataFrame | pd.Series | np.ndarray, num_coefs: np.ndarray, den_coefs: np.ndarray) -> pd.DataFrame | pd.Series | np.ndarray:
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
    

    return filtered_data

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


def get_blocks(data, well, max_na = 24, min_size = 24, divider_val = 'VSD power frequency'):

    data_well = data[data['Well Run'] ==well]

    ind = data_well[divider_val].index

    # Getting first and last days
    year1 = np.min(ind.year)
    month1 = np.min(ind[ind.year==year1].month)

    year2 = np.max(ind.year)
    month2 = np.max(ind[ind.year==year2].month)+1
    if month2 == 13:
        month2 = 1
        year2 += 1

    # Creating date range
    first = datetime.datetime(year1, month1, 1, 0, 0, 0)
    last = datetime.datetime(year2, month2, 1, 0, 0, 0)
    dates = pd.date_range(start=first, end=last, freq='H')
    days = pd.DataFrame(index = dates, data = np.zeros(dates.size))
    days.loc[ind,0] = 1

    # Dividing by groups
    division_dates = ind[ind.to_series().diff().dt.days>max_na]
    blcks = [None]*(division_dates.size+1)
    nblcs = 0

    if division_dates.size == 0:
        blcks[nblcs] = ind
        nblcs = 1

    else:
        for i in range(division_dates.size):
            
            if i>0:
                sub_ind =  ind[np.logical_and(ind<division_dates[i],ind>=division_dates[i-1])]
            else:
                sub_ind = ind[ind<division_dates[i]]
        
            if sub_ind.size > 0: 
                if (sub_ind[-1]-sub_ind[0]).days > min_size:
                    blcks[nblcs] = sub_ind
                    nblcs += 1

        sub_ind =  ind[ind>=division_dates[i]]
        if sub_ind.size > 0:
            if (sub_ind[-1]-sub_ind[0]).days > min_size:
                blcks[nblcs] = sub_ind
                nblcs += 1


    return blcks

def remove_outliers(data, nIQR = 1.5, diff = 0, compensate = False):

    # Transforming to dataframe
    if not (isinstance(data,pd.DataFrame) or isinstance(data,pd.Series)):
        vals = data.copy()
        data = pd.Series(vals)

    # Getting diff data
    diff_data = data.copy()
    for i in range(diff):
        diff_data = diff_data.diff()

    # Detecting outliers
    if diff > 0:
        Q1 = diff_data.quantile(q=0.25)
        Q3 = diff_data.quantile(q=0.75)
    else:
        Q1 = diff_data.abs().quantile(q=0.25)
        Q3 = diff_data.abs().quantile(q=0.75)



    IQR = Q3 - Q1


    outliers = (Q1 - nIQR*IQR > diff_data.abs()) |  (diff_data.abs() > Q3 + nIQR*IQR)
    outliers = outliers + outliers.shift(-1).fillna(False)
    
    no_outliers = (outliers == False)

    # new_outliers = outliers.copy()
    # while new_outliers.size > 0:
    #     non_outlier_data = data[no_outliers].diff()
    #     new_outliers = (Q1 - nIQR*IQR > non_outlier_data.abs()) |  (non_outlier_data.abs() > Q3 + nIQR*IQR)
    #     new_outliers = new_outliers + new_outliers.shift(1).fillna(False)
    #     new_outliers = new_outliers[new_outliers].index
    #     outliers[new_outliers] = True
    #     no_outliers[new_outliers] = False
    
    if compensate and diff == 1:
        diff_data[no_outliers] = 0
        data -= diff_data.cumsum()

    data = data[no_outliers]

    return data, no_outliers

def fill_gaps(data, fill_type = 'spline', grace_after = 24, resamp = '1h', min_size = 1):

    # Resampling data
    resampled_data = data.resample(resamp).max()

    # Finding NAs (gaps)
    resampled_nas =  resampled_data[resampled_data.isnull()]
    resampled_nas[:] = resampled_nas.index
    resampled_nas = resampled_nas.shift(1)


    # Getting higher than one hour interval
    borders = resampled_nas[resampled_nas.index.diff()>datetime.timedelta(hours=1)]
    intervals = [None]*borders.size

    # Finding intervals
    grace_after = datetime.timedelta(hours=grace_after)
    min_size = datetime.timedelta(hours=min_size)
    k = 0
    for i in range(borders.size):
        if i == borders.size-1:
            start = borders.index[i]
            end =  resampled_nas.index[-1]
            size =  end - start
        else:
            start = borders.index[i]
            end =  borders.values[i+1]
            size = end-start

        #if k > 0:
         #   if start-
        if size >= min_size:
            intervals[k] = (start, size+grace_after)
            k += 1
    
    del intervals[k:]

    filled_data = data.resample(resamp).interpolate()
    filled_data = filled_data.bfill()
    filled_data = filled_data.ffill()
    filled_data = filled_data.fillna(0)


    return filled_data, intervals

# def plot_recs(intervals,ax,data=None,ymax=0,ymin=0, add_pre=0):
    
#     if data is not None:
#         ymax = data.max()
#         ymin = data.min()
    
#     add_pre = datetime.timedelta(hours=add_pre)
#     size = len(intervals)
#     rect = [None]*size
#     for i in range(size):
#         rect[i] = Rectangle((intervals[i][0]-add_pre, ymin), intervals[i][1]+add_pre, (ymax-ymin), facecolor='lightgrey')
#         ax.add_patch(rect[i])

def get_MP(data: pd.DataFrame | pd.Series | np.ndarray, subseq_size: int, quartile: float = 0., ignore_start: bool = True) -> np.ndarray:
    """ Returns the matrix profile of each dimension of a time series.

        Returns the matrix profile of a time series. The matrix profile is defined by Prof. Eamon Keogh as the minimum distance profile for each subsequence in a time series. It is calculated here via the stumpy library.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Time series whose matrix profile is to be calculated.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        subseq_size: int
            Size of the subsequences to be analyzed in the matrix profile
        quartile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_start: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.

        Returns
        -------
        MP: numpy.ndarray
            Array with the matrix profile of the num_dim signals given.
            MP[:num_dim,0] -> each line corresponds to each signal's MP
            MP[num_dim:,1] -> each line corresponds to the closest match from each signal's subsequences
    """

    values, num_dim = _extract_values(data)
    series_size = values.shape[1]

    MP = np.zeros((2*num_dim,series_size-subseq_size+1),dtype=float)

    for k in range(num_dim):

        mp = stumpy.stump(values[k,:],subseq_size)

        if ignore_start:
            mp[:subseq_size,0] = 0
        
        if quartile != 0.:
            mp[:,0] -= np.quantile(mp[:,0],quartile)
            mp[:,0][mp[:,0]<0] = 0

        MP[k,:] = mp[:,0]
        MP[k+num_dim,:] = mp[:,1]

    return MP

def get_KDP(data: pd.DataFrame | np.ndarray, subseq_size: int, quartile: float = 0.75, ignore_start: bool = True, pre_calc_MP: np.ndarray = None) -> np.ndarray:
    """ Returns the k-dimensional profile of a time series.

        Returns all k-dimensional profiles of a time series. Each one is calculated as the maximum between each matrix profiles of every signal.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Time series whose matrix profile is to be calculated.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        subseq_size: int
            Size of the subsequences to be analyzed in the matrix profile
        quartile: float, optional
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_start: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.
        pre_calc_MP: numpy.ndarray, optional
            If MP of this time series have been calculated before from get_MP(), it can be given here to avoid repeated calculations

        Returns
        -------
        KDP: numpy.ndarray
            Array with the k-dimensional profile in ascending order.
            KDP[:num_dim,0] -> each line corresponds to each KDP -> [1-KDP, 2-KDP, ..., num_dim-KDP]
            KDP[num_dim:,1] -> each line corresponds to the signal of origin from each KDP.
    """

    if pre_calc_MP is None:
        MPs = get_MP(data, subseq_size, quartile=quartile, ignore_start=ignore_start)
    else:
        MPs = pre_calc_MP.copy()

    # Sorting MPS
    num_dim = MPs.shape[0]//2
    ind_kdp = np.argsort(MPs[:num_dim,:],axis=0)[::-1]

    KDP = np.empty((2*num_dim,MPs.shape[1]),dtype=float)

    KDP[:num_dim,:] = MPs[ind_kdp,np.arange(MPs.shape[1])[None,:]]
    KDP[num_dim:,:] = MPs[ind_kdp+num_dim,np.arange(MPs.shape[1])[None,:]]

    return KDP

def print_multiple(data: pd.DataFrame | pd.Series | np.ndarray, height: float = 1, width: float = 7, ylimits: list = None, xlabel: str = 'Timestamp', ylabel: str | list = None) -> Tuple[matplotlib.figure.Figure, list]: 

    values, num_dim = _extract_values(data)

    fig, axs = plt.subplots(num_dim, sharex=True, gridspec_kw={'hspace': 0},figsize=[width,height*num_dim])

    if isinstance(ylabel, str) and num_dim > 1:
        text = ylabel
        ylabel = [f'{text} {k}' for k in range(1,num_dim+1)]

    if ylabel is None and isinstance(data,pd.DataFrame):
        ylabel = data.columns

    xlimits = [0, values.shape[1]]
    if ylimits is None:
        ylimits = [0,np.max(values)]

    
    # Making axs a list, so nothing breaks
    if num_dim == 1:
        axs = [axs]
    
    if isinstance(data, np.ndarray):
        for k in range(num_dim):
            axs[k].plot(values[k,:])
            axs[k].set_ylim(ylimits)
            axs[k].grid(True)
            axs[k].set_ylabel(ylabel[k])
            axs[k].set_xlim(xlimits)
            axs[k].set_xlabel(xlabel)

    else:
        for k in range(num_dim):
            axs[k].plot(data.index,values[k,:])
            axs[k].set_ylim(ylimits)
            axs[k].grid(True)
            axs[k].set_ylabel(ylabel[k])
            axs[k].set_xlabel(xlabel)


    return fig, axs