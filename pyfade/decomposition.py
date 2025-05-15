import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import pywt

from typing import Literal, Tuple, Union, Any


def get_signal_decomp(data: Union[pd.DataFrame, pd.Series, np.ndarray], wavelet: str = 'haar', create_plot: bool = False, level: int = 1, height: float = 1, width: float = 7, **kwargs) -> Any:
    """ Perform a multi-level discrete wavelet decomposition. Returns recomposed signals at each level and coefficients.

        Performs a multi-level discrete wavelet decompostion on a signal. Returns the resulting signal from each decomposition level (approximation and detail) and their respective coefficients.

        Parameters
        ----------
        data: pandas.DataFrame or pandas.Series or numpy.ndarray
            Time series whose decomposition is performed.
            In case of multi-dimensional time series given as a numpy ndarray, its dimensions must be:
            data.size == (num_dim, series_size)
        wavelet: str, optional (default = 'haar')
            Mother wavelet. Uses the nomenclature from the pywavelet library.
        create_plot: bool, optional (default False)
            Plots each step of the decomposition if true.
        level: int, optional (default = 1)
            Level of the decomposition.
        height: float, optional (default = 7)
            Height of each plot window.
        width: float, optional (default = 7)
            Width of each plot window.

        Returns
        -------
        decomp_sig: list
            List of approximation and detail reconstructed signals at each step:

            decomp_sig = ( (sig_A_1, sig_D_1), (sig_A_2, sig_D_2), ...,  (sig_A_n, sig_D_n))

        decomp_coef: list
            List of approximation and detail coefficients at each step:

            decomp_coef = ( (coef_A_1, coef_D_1), (coef_A_2, coef_D_2), ..., (coef_A_n, coef_D_n))
    """

    # Getting index and value
    if isinstance(data, np.ndarray):
        values = data
        if values.ndim == 1:
            values = values[np.newaxis,:]
        index = np.arange(values.shape[1])
    elif isinstance(data,pd.Series):
        values = data.values
        values = values[np.newaxis,:]
        index = data.index
    elif isinstance(data,pd.DataFrame):
        values = data.values.T
        index = data.index

    style = kwargs.get('style','-b')
    markersize = kwargs.get('markersize',2)
    linewidth = kwargs.get('linewidth',1)


    # Creating variables
    decomp_sig = [None]*level
    decomp_coef = [None]*level
    cA = values[0,:].copy()

    # Calculating each level
    for lvl in range(level):

        # Performing discrete wavelet transform
        cA, cD = pywt.dwt(cA.copy(), wavelet=wavelet)

        # Coefficients of this transform
        coef_A = [cA, None] + [None]*lvl
        coef_D = [None, cD] + [None]*lvl

        # Reconstructing both signals
        sig_A = pywt.waverec(coef_A, wavelet=wavelet)
        sig_D = pywt.waverec(coef_D, wavelet=wavelet)

        # Getting reconstructed index
        if isinstance(data, np.ndarray) or not isinstance(index,pd.DatetimeIndex):
            new_ind_A = np.interp(np.linspace(0,1,sig_A.size),np.linspace(0,1,index.size),index)
            new_ind_D = np.interp(np.linspace(0,1,sig_D.size),np.linspace(0,1,index.size),index)

        else:
            new_ind_A = pd.date_range(start=index.min(), end=index.max(), periods=sig_A.size)
            new_ind_D = pd.date_range(start=index.min(), end=index.max(), periods=sig_D.size)

        # Writing signal and coefficient
        decomp_sig[lvl] = [pd.Series(sig_A,index=new_ind_A), pd.Series(sig_D,index=new_ind_D)]
        decomp_sig[lvl][0].name = f'Approximation {lvl+1}'
        decomp_sig[lvl][1].name = f'Detail {lvl+1}'
        decomp_coef[lvl] = (cA, cD)



    # Plot data if True
    if create_plot:

        # Plot original data
        plt.figure(figsize=(width,(level+1)*height))
        y_windows = level+1
        num_plots = 2*level+1
        axs = [None]*num_plots
        axs[0] = plt.subplot(y_windows,1,1)
        axs[0].grid(True)
        axs[0].plot(index,values[0,:],style,markersize=markersize,linewidth=linewidth)
        axs[0].set_ylabel('Original data')

        # Plot each level
        for lvl in range(level):

            # Approximation
            ax = plt.subplot(y_windows,2,2*(lvl+2)-1)
            axs[2*lvl+1] = ax
            if isinstance(data, np.ndarray):
                ind = decomp_sig[lvl][0][0]
                sig = decomp_sig[lvl][0][1]
                ax.plot(ind,sig,style,markersize=markersize,linewidth=linewidth)
                ax.grid(True)
                ax.set_ylabel(f'Approximation {lvl+1}')
            else:
                sig = decomp_sig[lvl][0]
                ax.plot(sig,style,markersize=markersize,linewidth=linewidth)
                ax.grid(True)
                ax.set_ylabel(f'Approximation {lvl+1}')

            # Detail
            ax = plt.subplot(y_windows,2,2*(lvl+2))
            axs[2*lvl+2] = ax
            if isinstance(data, np.ndarray):
                ind = decomp_sig[lvl][1][0]
                sig = decomp_sig[lvl][1][1]
                ax.plot(ind,sig,style,markersize=markersize,linewidth=linewidth)
                ax.grid(True)
                ax.set_ylabel(f'Detail {lvl+1}')
            else:
                sig = decomp_sig[lvl][1]
                ax.plot(sig,style,markersize=markersize,linewidth=linewidth)
                ax.grid(True)
                ax.set_ylabel(f'Detail {lvl+1}')


    return decomp_sig, decomp_coef