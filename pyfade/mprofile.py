"""
Functions for calculating distance profiles and matrix profiles of time series
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import stumpy

from typing import Union, Any

from pyfade import plot_multiple, get_signal_decomp

def wavelet_MP_from_KDP(data: Union[pd.DataFrame, pd.Series], subseq_size: int, dimension: int, **kwargs) -> pd.DataFrame:

    if isinstance(data,pd.Series):
        return wavelet_KDP(data,subseq_size,dimension,**kwargs)
    
    else:

        num_dim = data.columns.size

        MP = [None]*num_dim

        for k in range(num_dim):

            # Getting values
            MP[k] = wavelet_KDP(data.iloc[:,k],subseq_size,dimension,**kwargs)



    return MP


def wavelet_KDP(data: Union[pd.DataFrame, pd.Series], subseq_size: int, dimension: int, **kwargs) -> pd.DataFrame:
    """ Returns the K-dimensional profile of a multi-level wavelet decomposition, with a given dimension.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series
            Time series whose matrix profile is to be calculated.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        subseq_size: int
            Size of the subsequences to be analyzed in the matrix profile
        dimension: int
            Dimension of K-dimensional profile to be returned
        **kwargs:
            Optional inputs for get_MP_from_wavelets().


        Returns
        -------
        MP: list
            List with dataframe of each matrix profile. Each matrix profile dataframe has two columns:
            "data" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences

        See also:
        get_MP_from_wavelets():
            Calculates the matrix profile of wavelets from a multi-level wavelet decomposition of signal.
    """

    if not hasattr(kwargs,'level'):
        level = dimension

    if not hasattr(kwargs,'on_signal'):
        on_signal = True

    # Checking maximum level
    max_level = np.log2(subseq_size/3)
    if not on_signal and level > max_level:
        max_level = int(max_level)
        warnings.warn(f"Level given is higher than maximum, lowering it to {max_level}.")
        level = max_level
        dimension = max_level


    # Calculating Matrix Profile
    MP,_ = get_MP_from_wavelets(data,subseq_size,**kwargs)

    # Calculating K-DP
    KP = get_KDP(data, subseq_size, pre_calc_MP=MP)
    KP = KP[dimension-1]

    return KP


def get_MP_from_wavelets(data: Union[pd.DataFrame, pd.Series], subseq_size: int, level: int = 1, on_signals: bool = True, create_plot: bool = False, plot_data: bool = False, **kwargs) -> Any:
    """ Returns the matrix profile of a multi-level wavelet decomposition.

        Returns the matrix profile of a time series. The matrix profile is defined by Prof. Eamon Keogh as the minimum distance profile for each subsequence in a time series. It is calculated here via the stumpy library. The wavelet decomposition is done with the pywavelet library.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series
            Time series whose matrix profile is to be calculated.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        subseq_size: int
            Size of the subsequences to be analyzed in the matrix profile
        level: int, optional (default = 1)
            Level of the decomposition.
        on_signals: bool, optional (default = True)
            Selects whether the matrix profile will be calculated on the reconstructed signals or directly on the coefficients. Note that each level decreases the number of coefficients roughtly by 2.
        create_plot: bool, optional (default False)
            Plots each step matrix profile if true.
        plot_data: bool, optional (default False)
            Plots the approximation and detail data from the decomposition if true.
        **kwargs:
            Optional inputs for pyfade.get_MP(), pyfade.plot_multiple() and get_signal_decomp() functions.


        Returns
        -------
        MP: list
            List with dataframe of each matrix profile. Each matrix profile dataframe has two columns:
            "<data_name>" -> corresponding the value of the matrix profile.
            "match" -> corresponding to the closest match from each signal's subsequences.

        See also
        --------

        get_MP():
            function to calculate Matrix Profile

        plot_multiple():
            plot multiple curves

        get_signal_decomp():
            decomposes the signal with multi-level wavelet decomposition

    """


    # Checking maximum level
    max_level = np.log2(subseq_size/3)
    if not on_signals and level > max_level:
        max_level = int(max_level)
        warnings.warn(f"Level given is higher than maximum, lowering it to {max_level}.")
        level = max_level
    
    
    # Getting index and decomposition
    index = data.index
    signals, coefs = get_signal_decomp(data, level=level, **kwargs)

    # Creating MP list
    MPs = [None]*(level+1)

    if plot_data:
        data_plot = [None]*(level+1)

    if on_signals:

        # Detail Matrix Profile
        for lvl in range(level):

            cur_sig = signals[lvl][1]
            
            MPs[lvl] = get_MP(cur_sig,subseq_size,**kwargs)[0]
            MPs[lvl].name = f'Level {lvl+1}'
            if plot_data:
                data_plot[lvl] = cur_sig


        # Approximation Matrix Profile
        cur_sig = signals[level-1][0]

        MPs[level] = get_MP(cur_sig,subseq_size,**kwargs)[0]
        MPs[level].name = f'Level {level+1}'
        if plot_data:
            data_plot[level] = cur_sig

    else:

        # Detail Matrix Profile
        for lvl in range(level):

            cur_coef = coefs[lvl][1]
            subseq_size = subseq_size//2

            if isinstance(index,pd.DatetimeIndex):
                new_index = pd.date_range(start=index.min(), end=index.max(), periods=cur_coef.size)[:-subseq_size+1]
            else:
                new_index = np.linspace(index.min(), index.max(), cur_coef.size)[:-subseq_size+1]

            MPs[lvl] = get_MP(cur_coef,subseq_size,**kwargs)[0]
            MPs[lvl].index = new_index
            if plot_data:
                cur_sig = signals[lvl][1]
                data_plot[lvl] = cur_sig


        # Approximation Matrix Profile
        cur_coef = coefs[level-1][0]

        if isinstance(index,pd.DatetimeIndex):
            new_index = pd.date_range(start=index.min(), end=index.max(), periods=cur_coef.size)[:-subseq_size+1]
        else:
            new_index = np.linspace(index.min(), index.max(), cur_coef.size)[:-subseq_size+1]

        MPs[level] = get_MP(cur_coef,subseq_size,**kwargs)[0]
        MPs[level].index = new_index
        if plot_data:
            cur_sig = signals[level-1][0]
            data_plot[level] = cur_sig

    if create_plot:

        fig, axs = plot_multiple(MPs, **kwargs)

        if plot_data:
            plot_multiple(data_plot,fig=fig,axs=axs, **kwargs)


        return MPs, (fig,axs)


    return MPs, None

def get_MP(data: pd.DataFrame | pd.Series | np.ndarray | list, subseq_size: int, quantile: float = 0., ignore_extremes: bool = True, **kwargs) -> list:
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
        quantile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quantile
        ignore_extremes: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.

        Returns
        -------
        MP: list
            List with dataframe of each matrix profile. Each matrix profile dataframe has two columns:
            "<data_name>" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences
    """

    # Extracting relevant information depending on the data type
    if isinstance(data,pd.Series):
        index = data.index[:-subseq_size+1]
        values = data.values
        values = values[np.newaxis,:]
        num_dim = 1
        name = [data.name]
    elif isinstance(data,pd.DataFrame):
        index = data.index[:-subseq_size+1]
        values = data.values.T
        num_dim = values.shape[0]
        name = data.columns
    elif isinstance(data,np.ndarray):
        values = data
        if values.ndim == 1:
            values = values[np.newaxis,:]
        num_dim = values.shape[0]
        name = [f'data {k}' for k in range(num_dim)]
        index = None
    elif isinstance(data, list):
        num_dim = len(data)
        for k in range(num_dim):
            if not isinstance(data, pd.Series):
                raise Exception('Wrong type for data input. Must be dataframe, series, array or list of series!')
        name = [f'data {k}' for k in range(num_dim)]
        index = None
    else:
        raise Exception('Wrong type for data input. Must be dataframe, series, array or list of series!')


    MP = [None]*num_dim


    for k in range(num_dim):

        # Getting values
        if isinstance(data, list):
            val = data[k].values
            index = data[k].index[:-subseq_size+1]
            
        else:
            val = values[k,:]

        # Calculating MP
        mp = stumpy.stump(val,subseq_size,normalize=True)

        # Removing start
        if ignore_extremes:
            mp[:subseq_size,0] = 0
            mp[-subseq_size:,0] = 0
        
        # Removing quantile
        if quantile != 0.:
            mp[:,0] -= np.quantile(mp[:,0],quantile)
            mp[:,0][mp[:,0]<0] = 0

        MP[k] = pd.DataFrame(data={f'{name[k]}': mp[:,0], 'match': mp[:,1]}, index=index)

    return MP

def get_KDP(data: pd.DataFrame, subseq_size: int, pre_calc_MP: np.ndarray = None, num_dims: int = None, **kwargs) -> list:
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
        pre_calc_MP: list, optional
            If MP of this time series have been calculated before from get_MP(), it can be given here to avoid repeated calculations
        num_dims: int, optional
            Maximum number of dimensions to calculate K-dimensional profile. If not given, returns all.
        **kwargs
            Optional inputs that are passed to pyfade.get_MP() function.


        Returns
        -------
        KDP: list
            List with a dataframe of each k-dimensional profile, in ascending order. Each k-dimensional profile dataframe has two columns:
            "<k>-DP" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences

        See also
        ---------
        get_MP():
            function that calculates matrix profile

    """

    if pre_calc_MP is not None:
        
        MPs = pre_calc_MP

    else:
        
        MPs = get_MP(data, subseq_size, **kwargs)

    # Getting number of dimensions
    if num_dims is None:
        num_dims = len(MPs)

    # Preparing
    if isinstance(MPs[0].index, pd.DatetimeIndex):
        MPs_data = pd.concat([df.iloc[:,0].resample('h').nearest() for df in MPs],axis=1)
        MPs_match = pd.concat([df.iloc[:,1].resample('h').nearest() for df in MPs],axis=1)
    else:
        MPs_data = pd.concat([df.iloc[:,0] for df in MPs],axis=1)
        MPs_match = pd.concat([df.iloc[:,1] for df in MPs],axis=1)

    # Sorting MPS
    index = MPs_data.index
    ind_kdp = np.argsort(MPs_data,axis=1)[:,::-1][:,:num_dims]

    # creating KDP
    KDP = [None]*num_dims
    for k in range(num_dims):
        value = MPs_data.values[np.arange(index.size),ind_kdp[:,k]]
        match = MPs_match.values[np.arange(index.size),ind_kdp[:,k]] 
        KDP[k] = pd.DataFrame(data={f'{k+1}-DP': value, 'match': match}, index=index)


    return KDP