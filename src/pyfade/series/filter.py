"""
Functions for filtering data.
"""

import numpy as np
import scipy

from scipy.linalg import pascal
from scipy.signal import butter
from enum import Enum

from abc import ABC, abstractmethod

import scipy.signal

class FilterType(Enum):
    """ Filter selections """
     
    MA = 0, "Moving average filter"
    DIFF = 1, "Finite difference filter"
    BUTTER_LOW = 2, "Low-pass butterworth filter"
    BUTTER_HIGH = 3, "High-pass butterworth filter"
    BUTTER_PASS = 4, "Band-pass butterworth filter"
    

class DataFilterInterface(ABC):
    """
    Abstract class that holds a filtering technique.

    This class is used to define the interface for different filtering techniques that can be used
    in the DataFrame class.

    The filters are define by their difference equations. Each filter differs by the coefficients of the 
    numerator and denominator.

    Properties
    ----------
    numerator_coeffs: np.ndarray
        Array that contains the coefficients of the numerators of the filter.
    denominator_coeffs: np.ndarray
        Array that contains the coefficients of the denominators of the filter.

    Methods
    -------
    apply_filter(data):
        Applies the filter inplace on the given data.

    See Also
    --------
    FilterType: Enum for filter types.
    DataSet: Base class for datasets that holds the data and index.
    """

    @property
    @abstractmethod
    def numerator_coeffs(self) -> np.ndarray:
        """
        np.ndarray: Coefficients of the numerator of the difference equation.
        """

        pass

    @property
    @abstractmethod
    def denominator_coeffs(self) -> np.ndarray:
        """
        np.ndarray: Coefficients of the denominator of the difference equation.
        """

        pass

    def apply_filter(self, data: np.ndarray):
        
        filtered_data = scipy.signal.lfilter(self.numerator_coeffs, self.denominator_coeffs, data)
        data[:] = filtered_data

class MovingAverageFilter(DataFilterInterface):
    """ 
    Class that sets up a Moving Average filter.

    Creates a Moving Average filter model, returning the coeficients from the numerator and denominator 
    of its difference equation. Applying this filter to a time series x_i corresponds to:

    y_i = (x_i + x_{i-1} + ... + x_{i-lag}) / (lag+1)

    Parameters
    ----------
    lag: int
        Lag of the moving average filter.
    """

    numerator_coeffs: np.ndarray = None
    denominator_coeffs: np.ndarray = None

    def __init__(self, lag: int):
        self.denominator_coeffs = np.ones(1,dtype=float)
        self.numerator_coeffs   = np.ones(lag, dtype=float) / (lag+1)

class FiniteDifferenceFilter(DataFilterInterface):
    """ 
    Calculates the difference equation coeficients for a backward finite difference filter, without the denominator.

    Creates a backward finite difference filter model, returning the coeficients from the numerator and 
    denominator of its difference equation. Applying this filter to a time series x_i corresponds to:

    order = 1:
        y_i = x_i-x_{i-1}

    order = 2:
        y_i = x_i - 2*x_{i-1} + x_{i-2}

    Parameters
    ----------
    order: int
        Order of the difference.
    """
    
    denominator_coeffs: np.ndarray = None
    numerator_coeffs: np.ndarray = None

    def __init__(self, order: int):
        self.denominator_coeffs = np.ones(1, dtype=float)
        self.numerator_coeffs   = pascal(order+1, kind='lower')[order, :order+2] * (-1)**np.arange(order+1)

class ButterFilter(DataFilterInterface):
    
    denominator_coeffs: np.ndarray = None
    numerator_coeffs: np.ndarray = None

    def __init__(self, method:FilterType, order: int, cutoff_freq: float|list, sampling_freq: float):
        
        if method == FilterType.BUTTER_LOW:
            self.numerator_coeffs, self.denominator_coeffs = butter(order, cutoff_freq,fs=sampling_freq,btype='low')
        elif method == FilterType.BUTTER_HIGH:
            self.numerator_coeffs, self.denominator_coeffs = butter(order, cutoff_freq,fs=sampling_freq,btype='high')
        elif method == FilterType.BUTTER_PASS:
            self.numerator_coeffs, self.denominator_coeffs = butter(order, cutoff_freq,fs=sampling_freq,btype='pass')
        

def get_filter(FilterType: FilterType,
               *args,
               **kwargs) -> DataFilterInterface:
    """ Returns the difference equation coefficients of a filter.

        Returns the matrix profile of a time series. The matrix profile is defined by Prof. Eamon Keogh as the minimum distance profile for each subsequence in a time series. It is calculated here via the stumpy library.

        Parameters
        ----------
        FilterType: FilterType enum
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

    match FilterType:
        
        case FilterType.MA:
            
            err = ValueError("Moving Average filter requires an int lag input.")
            if len(args)==0:
                raise err
            
            lag = args[0]
            if not isinstance(lag,int):
                raise err
            
            return MovingAverageFilter(lag)
        
        case FilterType.DIFF:

            err = ValueError("Finite Difference filter requires an int order input.")
            if len(args)==0:
                raise err
            
            order = args[0]
            if not isinstance(order,int):
                raise err
            
            return FiniteDifferenceFilter(order)


        case FilterType.BUTTER_LOW | FilterType.BUTTER_HIGH:

            if len(args) < 2:
                raise ValueError("Low or High Butter filter requires two int inputs, order and cutoff frequency.")
            order = args[0]
            cutoff_freq = args[1]

            sampling_freq = kwargs.get("sampling_freq",1)

            return ButterFilter(FilterType,order,cutoff_freq,sampling_freq)
            
        case FilterType.BUTTER_PASS:

            if len(args) < 2:
                raise ValueError("Pass Butter filter requires two inputs: order (int) and cutoff frequency ([int,int]).")
            order = args[0]
            cutoff_freq = args[1]

            sampling_freq = kwargs.get("sampling_freq",1)

            return ButterFilter(FilterType,order,cutoff_freq,sampling_freq)
            
        case _:
            raise AssertionError('Filter type not defined!')
