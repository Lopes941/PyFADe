import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import stumpy

from typing import Union

from .preprocessing import _extract_values
from .plot import print_multiple


def get_MP_from_wavelets(data: Union[pd.DataFrame, pd.Series], subseq_size: int, quartile: float = 0., ignore_start: bool = True, level: int = 1, wavelet: str = 'haar', on_signals: bool = True, create_plot: bool = True, plot_data: bool = False) -> np.ndarray:

    from .signals import get_signal_decomp

    
    # Getting index and decomposition
    index = data.index
    signals, coefs = get_signal_decomp(data, wavelet=wavelet, create_plot=False, level = level)

    # Creating MP list
    MPs = [None]*(level+1)

    if plot_data:
        data_plot = [None]*(level+1)

    if on_signals:

        # Detail Matrix Profile
        for lvl in range(level):

            cur_sig = signals[lvl][1]
            new_index = cur_sig.index[:-subseq_size+1]

            mp = get_MP(cur_sig,subseq_size,quartile=quartile,ignore_start=ignore_start)[0,:]
            MPs[lvl] = pd.Series(mp, index=new_index)
            if plot_data:
                data_plot[lvl] = cur_sig


        # Approximation Matrix Profile
        cur_sig = signals[level-1][0]
        index = cur_sig.index[:-subseq_size+1]

        mp = get_MP(cur_sig,subseq_size,quartile=quartile,ignore_start=ignore_start)[0,:]
        MPs[level] = pd.Series(mp,index=new_index)
        if plot_data:
            data_plot[level] = cur_sig

    else:

        # Detail Matrix Profile
        for lvl in range(level):

            cur_coef = coefs[lvl][1]
            subseq_size = subseq_size//2
            new_index = pd.date_range(start=index.min(), end=index.max(), periods=cur_coef.size)[:-subseq_size+1]

            mp = get_MP(cur_coef,subseq_size,quartile=quartile,ignore_start=ignore_start)[0,:]
            MPs[lvl] = pd.Series(mp, index=new_index)
            if plot_data:
                cur_sig = signals[lvl][1]
                data_plot[lvl] = cur_sig


        # Approximation Matrix Profile
        cur_coef = coefs[level-1][0]
        new_index = pd.date_range(start=index.min(), end=index.max(), periods=cur_coef.size)[:-subseq_size+1]

        mp = get_MP(cur_coef,subseq_size,quartile=quartile,ignore_start=ignore_start)[0,:]
        MPs[level] = pd.Series(mp,index=new_index)
        if plot_data:
            cur_sig = signals[level-1][0]
            data_plot[level] = cur_sig

    if create_plot:

        fig, ax = print_multiple(MPs)

        if plot_data:
            print_multiple(data_plot,fig=fig,axs=ax)


    return MPs

# def get_MP_list(data: list, subseq_size: int, quartile: float = 0., ignore_start: bool = True) -> np.ndarray:

#     num_dim = len(data)


def get_MP(data: pd.DataFrame | pd.Series | np.ndarray | list, subseq_size: int, quartile: float = 0., ignore_start: bool = True) -> np.ndarray:
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

    if isinstance(data,list):
        raise Exception('Not implemented!')
    else:

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

def get_KDP(data: Union[pd.DataFrame, np.ndarray], subseq_size: int, quartile: float = 0.75, ignore_start: bool = True, pre_calc_MP: np.ndarray = None) -> np.ndarray:
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