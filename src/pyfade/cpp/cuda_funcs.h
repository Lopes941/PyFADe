#pragma once

#include <cstddef>
#include <cufft.h>

void launch_multiply_complex(cufftDoubleComplex* d_fft_series, 
                                const cufftDoubleComplex* d_fft_kernel, 
                                size_t fft_size);

void launch_distance_profile(const double* series, 
                                    const double* QT,
                                    double* d_QT_first,
                                    const double* means, 
                                    const double* stds, 
                                    double* MP, 
                                    size_t* inds_MP, 
                                    const size_t i, 
                                    const size_t final_size, 
                                    const size_t interval_size,
                                    const size_t exclusion_zone_size,
                                    const size_t initial_size,
                                    const bool left_only);

void launch_STOMP_iterations(const double* series, 
                        double* QT_even,
                        double* QT_odd, 
                        const double* means, 
                        const double* stds, 
                        double* MP, 
                        size_t* inds_MP, 
                        const double* QT_first, 
                        size_t& i, 
                        const size_t final_size, 
                        const size_t interval_size,
                        const size_t exclusion_zone_size,
                        const size_t initial_size,
                        const bool left_only);