"""
Functions for calculating distance profiles and matrix profiles of time series
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import stumpy

from typing import Union, Any

<<<<<<< HEAD
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
=======
from pyfade import plot_multiple

def wavelet_KDP(data: Union[pd.DataFrame, pd.Series], subseq_size: int, dimension: int, quartile: float = 0., ignore_extremes: bool = True, level: int = 1, wavelet: str = 'haar', on_signals: bool = True, create_plot: bool = False, plot_data: bool = False, height: float = 7, width: float = 1):
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
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
<<<<<<< HEAD
        **kwargs:
            Optional inputs for get_MP_from_wavelets().
=======
        quartile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_extremes: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.
        level: int, optional (default = 1)
            Level of the decomposition.
        wavelet: str, optional (default = 'haar')
            Mother wavelet. Uses the nomenclature from the pywavelet library.
        on_signals: bool, optional (default = True)
            Selects whether the matrix profile will be calculated on the reconstructed signals or directly on the coefficients. Note that each level decreases the number of coefficients roughtly by 2.
        create_plot: bool, optional (default False)
            Plots each step matrix profile if true.
        plot_data: bool, optional (default False)
            Plots the approximation and detail data from the decomposition if true.
        height: float, optional (default = 7)
            Height of each plot window.
        width: float, optional (default = 1)
            Width of each plot window.
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca


        Returns
        -------
        MP: list
            List with dataframe of each matrix profile. Each matrix profile dataframe has two columns:
            "data" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences
<<<<<<< HEAD

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
=======
    """

    # Calculating Matrix Profile
    MP, _ = get_MP_from_wavelets(data,subseq_size,quartile=quartile,ignore_extremes=ignore_extremes,level=level,wavelet=wavelet,on_signals=on_signals,create_plot=create_plot,plot_data=plot_data,height=height,width=width)

    # Calculating K-DP
    KP,_ = get_KDP(data, subseq_size, pre_calc_MP=MP)
    KP = KP.iloc[:,dimension-1]
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca

    return KP


<<<<<<< HEAD
def get_MP_from_wavelets(data: Union[pd.DataFrame, pd.Series], subseq_size: int, level: int = 1, on_signals: bool = True, create_plot: bool = False, plot_data: bool = False, **kwargs) -> Any:
=======
def get_MP_from_wavelets(data: Union[pd.DataFrame, pd.Series], subseq_size: int, quartile: float = 0., ignore_extremes: bool = True, level: int = 1, wavelet: str = 'haar', on_signals: bool = True, create_plot: bool = False, plot_data: bool = False, height: float = 7, width: float = 1) -> Any:
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
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
<<<<<<< HEAD
        level: int, optional (default = 1)
            Level of the decomposition.
=======
        quartile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_extremes: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.
        level: int, optional (default = 1)
            Level of the decomposition.
        wavelet: str, optional (default = 'haar')
            Mother wavelet. Uses the nomenclature from the pywavelet library.
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        on_signals: bool, optional (default = True)
            Selects whether the matrix profile will be calculated on the reconstructed signals or directly on the coefficients. Note that each level decreases the number of coefficients roughtly by 2.
        create_plot: bool, optional (default False)
            Plots each step matrix profile if true.
        plot_data: bool, optional (default False)
            Plots the approximation and detail data from the decomposition if true.
<<<<<<< HEAD
        **kwargs:
            Optional inputs for pyfade.get_MP(), pyfade.plot_multiple() and get_signal_decomp() functions.
=======
        height: float, optional (default = 7)
            Height of each plot window.
        width: float, optional (default = 1)
            Width of each plot window.
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca


        Returns
        -------
        MP: list
            List with dataframe of each matrix profile. Each matrix profile dataframe has two columns:
<<<<<<< HEAD
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
=======
            "<data_name>" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences
    """

    from .decomposition import get_signal_decomp


    # Checking maximum level
    max_level = np.log2(subseq_size/3)
    if level > max_level:
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        max_level = int(max_level)
        warnings.warn(f"Level given is higher than maximum, lowering it to {max_level}.")
        level = max_level
    
    
    # Getting index and decomposition
    index = data.index
<<<<<<< HEAD
    signals, coefs = get_signal_decomp(data, level=level, **kwargs)
=======
    signals, coefs = get_signal_decomp(data, wavelet=wavelet, create_plot=False, level = level)
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca

    # Creating MP list
    MPs = [None]*(level+1)

    if plot_data:
        data_plot = [None]*(level+1)

    if on_signals:

        # Detail Matrix Profile
        for lvl in range(level):

            cur_sig = signals[lvl][1]
<<<<<<< HEAD
            
            MPs[lvl] = get_MP(cur_sig,subseq_size,**kwargs)[0]
=======

            MPs[lvl] = get_MP(cur_sig,subseq_size,quartile=quartile,ignore_extremes=ignore_extremes)[0]
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
            MPs[lvl].name = f'Level {lvl+1}'
            if plot_data:
                data_plot[lvl] = cur_sig


        # Approximation Matrix Profile
        cur_sig = signals[level-1][0]

<<<<<<< HEAD
        MPs[level] = get_MP(cur_sig,subseq_size,**kwargs)[0]
=======
        MPs[level] = get_MP(cur_sig,subseq_size,quartile=quartile,ignore_extremes=ignore_extremes)[0]
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        MPs[level].name = f'Level {level+1}'
        if plot_data:
            data_plot[level] = cur_sig

    else:

        # Detail Matrix Profile
        for lvl in range(level):

            cur_coef = coefs[lvl][1]
            subseq_size = subseq_size//2

<<<<<<< HEAD
            if isinstance(index,pd.DatetimeIndex):
=======
            if isinstance(index,pd.Timestamp):
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
                new_index = pd.date_range(start=index.min(), end=index.max(), periods=cur_coef.size)[:-subseq_size+1]
            else:
                new_index = np.linspace(index.min(), index.max(), cur_coef.size)[:-subseq_size+1]

<<<<<<< HEAD
            MPs[lvl] = get_MP(cur_coef,subseq_size,**kwargs)[0]
=======
            MPs[lvl] = get_MP(cur_coef,subseq_size,quartile=quartile,ignore_extremes=ignore_extremes)[0]
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
            MPs[lvl].index = new_index
            if plot_data:
                cur_sig = signals[lvl][1]
                data_plot[lvl] = cur_sig


        # Approximation Matrix Profile
        cur_coef = coefs[level-1][0]

<<<<<<< HEAD
        if isinstance(index,pd.DatetimeIndex):
=======
        if isinstance(index,pd.Timestamp):
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
            new_index = pd.date_range(start=index.min(), end=index.max(), periods=cur_coef.size)[:-subseq_size+1]
        else:
            new_index = np.linspace(index.min(), index.max(), cur_coef.size)[:-subseq_size+1]

<<<<<<< HEAD
        MPs[level] = get_MP(cur_coef,subseq_size,**kwargs)[0]
=======
        MPs[level] = get_MP(cur_coef,subseq_size,quartile=quartile,ignore_extremes=ignore_extremes)[0]
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        MPs[level].index = new_index
        if plot_data:
            cur_sig = signals[level-1][0]
            data_plot[level] = cur_sig

    if create_plot:

<<<<<<< HEAD
        fig, axs = plot_multiple(MPs, **kwargs)

        if plot_data:
            plot_multiple(data_plot,fig=fig,axs=axs, **kwargs)
=======
        fig, axs = plot_multiple(MPs)

        if plot_data:
            plot_multiple(data_plot,fig=fig,axs=axs)
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca


        return MPs, (fig,axs)


    return MPs, None

<<<<<<< HEAD
def get_MP(data: pd.DataFrame | pd.Series | np.ndarray | list, subseq_size: int, quantile: float = 0., ignore_extremes: bool = True, **kwargs) -> list:
=======
def get_MP(data: pd.DataFrame | pd.Series | np.ndarray | list, subseq_size: int, quartile: float = 0., ignore_extremes: bool = True) -> list:
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
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
<<<<<<< HEAD
        quantile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quantile
=======
        quartile: float, optional (default = 0.)
            Moves the origin from 0. to a value defined by the value in this quartile
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        ignore_extremes: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.

        Returns
        -------
        MP: list
            List with dataframe of each matrix profile. Each matrix profile dataframe has two columns:
<<<<<<< HEAD
            "<data_name>" -> corresponding the value of the matrix profile
=======
            "data" -> corresponding the value of the matrix profile
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
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
<<<<<<< HEAD
        mp = stumpy.stump(val,subseq_size,normalize=True)
=======
        mp = stumpy.stump(val,subseq_size)
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca

        # Removing start
        if ignore_extremes:
            mp[:subseq_size,0] = 0
            mp[-subseq_size:,0] = 0
        
        # Removing quantile
<<<<<<< HEAD
        if quantile != 0.:
            mp[:,0] -= np.quantile(mp[:,0],quantile)
=======
        if quartile != 0.:
            mp[:,0] -= np.quantile(mp[:,0],quartile)
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
            mp[:,0][mp[:,0]<0] = 0

        MP[k] = pd.DataFrame(data={f'{name[k]}': mp[:,0], 'match': mp[:,1]}, index=index)

    return MP

<<<<<<< HEAD
def get_KDP(data: pd.DataFrame, subseq_size: int, pre_calc_MP: np.ndarray = None, num_dims: int = None, **kwargs) -> list:
=======
def get_KDP(data: pd.DataFrame, subseq_size: int, quartile: float = 0.75, ignore_extremes: bool = True, pre_calc_MP: np.ndarray = None, num_dims: int = None) -> list:
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
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
<<<<<<< HEAD
=======
        quartile: float, optional
            Moves the origin from 0. to a value defined by the value in this quartile
        ignore_extremes: bool, optional (default = True)
            Makes the matrix profile of the first subseq_size subsequences zero. Removes the boundary influence of filters.
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        pre_calc_MP: list, optional
            If MP of this time series have been calculated before from get_MP(), it can be given here to avoid repeated calculations
        num_dims: int, optional
            Maximum number of dimensions to calculate K-dimensional profile. If not given, returns all.
<<<<<<< HEAD
        **kwargs
            Optional inputs that are passed to pyfade.get_MP() function.


=======

            
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
        Returns
        -------
        KDP: list
            List with a dataframe of each k-dimensional profile, in ascending order. Each k-dimensional profile dataframe has two columns:
<<<<<<< HEAD
            "<k>-DP" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences

        See also
        ---------
        get_MP():
            function that calculates matrix profile

=======
            "data" -> corresponding the value of the matrix profile
            "match" -> corresponding to the closest match from each signal's subsequences
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
    """

    if pre_calc_MP is not None:
        
<<<<<<< HEAD
        MPs = pre_calc_MP

    else:
        
        MPs = get_MP(data, subseq_size, **kwargs)
=======
        MPs = pre_calc_MP.copy()

    else:
        
        MPs = get_MP(data, subseq_size, quartile=quartile, ignore_extremes=ignore_extremes)
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca

    # Getting number of dimensions
    if num_dims is None:
        num_dims = len(MPs)

    # Preparing
<<<<<<< HEAD
    if isinstance(MPs[0].index, pd.DatetimeIndex):
        MPs_data = pd.concat([df.iloc[:,0].resample('h').nearest() for df in MPs],axis=1)
        MPs_match = pd.concat([df.iloc[:,1].resample('h').nearest() for df in MPs],axis=1)
    else:
        MPs_data = pd.concat([df.iloc[:,0] for df in MPs],axis=1)
        MPs_match = pd.concat([df.iloc[:,1] for df in MPs],axis=1)
=======
    cols = np.array([df.columns[0] for df in MPs])

    if isinstance(MPs[0].index, pd.DatetimeIndex):
        MPs_data = pd.concat([df.iloc[:,0].resample('h').nearest() for df in MPs],axis=1)
    else:
        MPs_data = pd.concat([df.iloc[:,0] for df in MPs],axis=1)
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca

    # Sorting MPS
    index = MPs_data.index
    ind_kdp = np.argsort(MPs_data,axis=1)[:,::-1][:,:num_dims]
<<<<<<< HEAD

    # creating KDP
    KDP = [None]*num_dims
    for k in range(num_dims):
        value = MPs_data.values[np.arange(index.size),ind_kdp[:,k]]
        match = MPs_match.values[np.arange(index.size),ind_kdp[:,k]] 
        KDP[k] = pd.DataFrame(data={f'{k+1}-DP': value, 'match': match}, index=index)


    return KDP
=======
    columns = cols[ind_kdp]

    KDP = pd.DataFrame(np.take_along_axis(MPs_data.values,ind_kdp,axis=1), index=index)
    KDP.columns = [f'{k}-KDP' for k in range(1,num_dims+1)]

    return KDP, columns
>>>>>>> 6e4febcb05cc92ae2528ad59489a503fc5a19bca
