#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>


/**
 * @brief Runs the convolution from the data with its first window;
 * 
 * @param QT Convolution vector.
 * @param series_padded Dataset with padded zeros.
 * @param Q_padded Padded first window.
 * @param padded_size Total size of the padded data.
 * @param interval_size Window size.
 * @param mp_size Size of the matrix profile.
 */                     
void cpu_convolve(double *QT,
                const double* series_padded, 
                const double* Q_padded, 
                const int padded_size,
                const int interval_size,
                const int mp_size);