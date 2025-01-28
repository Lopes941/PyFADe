import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import scipy as sp
from scipy.signal import butter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from statsmodels.tsa.seasonal import MSTL, DecomposeResult
import stumpy
import re
import warnings
from enum import Enum

class Filter_type(Enum):
    BUTTER_LOW = 0
    BUTTER_HIGH = 1
    BUTTER_PASS = 2
    MA = 3
    DIFF = 4

def get_filter(filter_type: Filter_type, sampling_freq = 1, **kwargs) -> np.ndarray | np.ndarray:
    """
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


def _filter_ma(lag: int) -> np.ndarray | np.ndarray:
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

def _filter_diff(order: int) -> np.ndarray | np.ndarray:
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

def plot_recs(intervals,ax,data=None,ymax=0,ymin=0, add_pre=0):

    
    if data is not None:
        ymax = data.max()
        ymin = data.min()
    
    add_pre = datetime.timedelta(hours=add_pre)
    size = len(intervals)
    rect = [None]*size
    for i in range(size):
        rect[i] = Rectangle((intervals[i][0]-add_pre, ymin), intervals[i][1]+add_pre, (ymax-ymin), facecolor='lightgrey')
        ax.add_patch(rect[i])

def get_MP(data: pd.DataFrame | pd.Series | np.ndarray, subseq_size: int, quartile: float = 0., ignore_start: bool = True) -> np.ndarray:
    """ Returns the matrix profile of each dimension of a time series.

        Returns the matrix profile of a time series. The matrix profile is defined by Prof. Eamon Keogh as the minimum distance profile for each subsequence in a time series. It is calculated here via the stumpy library.

        Parameters
        ----------
        data: pandas.Series or numpy.ndarray
            Time series whose matrix profile is to be calculated
        subseq_size: int
            Size of the subsequences to be analyzed in the matrix profile
        quartile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_start: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.
        Returns
        -------
        mp: numpy.ndarray
            Array with the matrix profile. Same format as stumpy library.
            mp[:,0] -> matrix profile
            mp[:,1] -> matrix profile index
            mp[:,2] -> left matrix profile index
            mp[:,3] -> right matrix profile index
    """

    if isinstance(data,pd.Series):
        values = data.values
        num_dim = 1
    elif isinstance(data,pd.DataFrame):
        values = data.values
        num_dim = data.num_columns
    else:
        values = data.T
        num_dim = values.shape[1]

    MP = np.empty((2*num_dim,values.shape[0]-subseq_size+1),dtype=float)

    for k in range(num_dim):

        mp = stumpy.stump(values[:,k],subseq_size)

        if ignore_start:
            mp[subseq_size:,0] = 0
        
        if quartile != 0.:
            mp[:,0] -= np.quantile(mp[:,0],quartile)
            mp[:,0][mp[:,0]<0] = 0

        MP[k,:] = mp[:,0]
        MP[k+num_dim,:] = mp[:,1]

    return MP

def get_KDP(data: pd.DataFrame | np.ndarray, subseq_size: int, quartile: float = 0.75, ignore_start: bool = True) -> np.ndarray:
    """ Returns the k-dimensional profile of a multidimensional time series.

        Returns the matrix profile of a time series. The matrix profile is defined by Prof. Eamon Keogh as the minimum distance profile for each subsequence in a time series. It is calculated here via the stumpy library.

        Parameters
        ----------
        data: pandas.Series or numpy.ndarray
            Time series whose matrix profile is to be calculated
        subseq_size: int
            Size of the subsequences to be analyzed in the matrix profile
        quartile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_start: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.
        Returns
        -------
        mp: numpy.ndarray
            Array with the matrix profile. Same format as stumpy library.
            mp[:,0] -> matrix profile
            mp[:,1] -> matrix profile index
            mp[:,2] -> left matrix profile index
            mp[:,3] -> right matrix profile index
    """

    pass