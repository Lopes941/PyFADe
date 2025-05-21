#pragma once

#include <cstddef>
#include <cufft.h>

void launch_multiply_complex(cufftDoubleComplex* d_fft_series, 
                                const cufftDoubleComplex* d_fft_kernel, 
                                int fft_size);

void launch_iff_normalization(double* data, 
                            int size);


void launch_STOMP_iterations(const double* series, 
                            double* QT,
                            const double* means, 
                            const double* stds, 
                            double* MP, 
                            int* inds_MP, 
                            int &i, 
                            const int final_size, 
                            const int interval_size,
                            const int exclusion_zone_size,
                            const bool left_only);