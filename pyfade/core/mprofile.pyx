# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

import pkg_resources
import numpy as np
import cython

import cupy as cp
import cupyx.scipy.fft as cuda_fft
import scipy.fft as fft

cimport numpy 		as np
from libc.math 		cimport ceil

cdef int BLOCK_SIZE = 256
cdef double EXCLUSION_SIZE = 0.5

cpdef void STOMP_iterations(object series, object gpu_MP, object gpu_inds_MP, object QT_even, object means, object stds, size_t size, size_t interval_size, size_t skip_start, np.int16_t only_left):

    cdef size_t i
    cdef size_t final_size = size-interval_size+1
    cdef size_t GRID_SIZE = (final_size + BLOCK_SIZE - 1) // BLOCK_SIZE 

    cdef size_t exclusion_zone_size = <size_t> ceil(EXCLUSION_SIZE*interval_size)

    cubin_path = pkg_resources.resource_filename('pyfade', 'core/cuda_funcs.cubin')
    module = cp.RawModule(path=cubin_path)
    ker_sum = module.get_function('stomp_iteration')

    cdef object QT_odd = QT_even.copy()
    cdef object QT_first = QT_even.copy()

    for i in range(1,final_size):

        if i%2==0:
            ker_sum((GRID_SIZE,), (BLOCK_SIZE, ), (series, QT_even, QT_odd, means, stds, gpu_MP, gpu_inds_MP, QT_first, cp.int32(i), cp.int32(final_size), cp.int32(interval_size), cp.int32(exclusion_zone_size), cp.int32(skip_start), cp.int16(only_left)) )
        else:
            ker_sum((GRID_SIZE,), (BLOCK_SIZE, ), (series, QT_odd, QT_even, means, stds, gpu_MP, gpu_inds_MP, QT_first, cp.int32(i), cp.int32(final_size), cp.int32(interval_size), cp.int32(exclusion_zone_size), cp.int32(skip_start), cp.int16(only_left)) )

    gpu_MP[:skip_start] = 0


def get_matrix_profile(cpu_series: np.ndarray, interval_size: int, skip_start: int, only_left: bool) -> np.ndarray:


    series_size: int = cpu_series.size

    series: cp.ndarray = cp.asarray(cpu_series).astype(cp.float64)
    if interval_size > 1:
        sums: cp.ndarray = cp.concatenate([cp.zeros(1, dtype=series.dtype), cp.cumsum(series)]).astype(cp.float64)
        sums_sqr: cp.ndarray = cp.concatenate([cp.zeros(1, dtype=series.dtype), cp.cumsum(cp.square(series))]).astype(cp.float64)
        means: cp.ndarray = (sums[interval_size:]-sums[:-interval_size])/interval_size
        stds: cp.ndarray = cp.sqrt((sums_sqr[interval_size:]-sums_sqr[:-interval_size])/interval_size - cp.square(means))
    else:
        means: cp.ndarray = series.copy()
        stds: cp.ndarray = cp.zeros(series_size-interval_size+1, dtype=cp.float64)


    QT: cp.ndarray = cp.convolve(series,np.flip(series[:interval_size]),'valid')


    den = interval_size*stds[0]*stds
    den[cp.abs(den)<1e-9] = 1e-9
    D: cp.ndarray = cp.sqrt(2*interval_size * (1 - (QT - interval_size*means[0]*means)/(den)) ) 
    I: cp.ndarray = cp.zeros(series_size-interval_size+1, dtype=cp.int32)
    
    
    STOMP_iterations(series, D, I, QT, means, stds, series_size, interval_size, skip_start, only_left)
    
    cdef np.ndarray[double,ndim=2] MP = np.zeros((series_size-interval_size+1,2),dtype=float)

    MP[:,0] = cp.asnumpy(D)
    MP[:,1] = cp.asnumpy(I)

 
    return MP
