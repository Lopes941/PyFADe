#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>

#include <cstddef>
#include <cufft.h>
#include <memory>

/**
 * @brief Run the STOMP algorithm using CUDA.
 * 
 * Runs the matrix profile of a timeseries using CUDA. Automatically fills the MP and inds_MP
 * variables, which are changed in-place.
 * 
 * @param observed_dataset The dataset from which the matrix profile is calculated.
 * @param means The rolling mean of the dataset.
 * @param stds The rolling standard deviation of the dataset.
 * @param QT The convolution between the dataset and the window. Updated at each iteration.
 * @param MP The matrix profile vector that is changed inplace.
 * @param inds_MP The matrix profile index vector that is changed inplace.
 * @param start_location Integer that marks the starting index from this update.
 * @param final_size Total size of the matrix profile.
 * @param interval_size Window size.
 * @param exclusion_zone_size Size of the exclusion zone.
 * @param left_only Flag that indicates if only data to the left of the window is checked.
 */
void cuda_STOMP_iterations(const std::shared_ptr<cfade::DataSet> observed_dataset,
                        const std::shared_ptr<cfade::VectorGroup<double>> means,
                        const std::shared_ptr<cfade::VectorGroup<double>> stds,
                        double* QT,
                        std::shared_ptr<cfade::VectorGroup<double>> MP, 
                        std::shared_ptr<cfade::VectorGroup<int>> inds_MP,
                        int start_location, 
                        const int final_size, 
                        const int interval_size,
                        const int exclusion_zone_size,
                        const bool left_only);

/**
 * @brief Runs the convolution from the data with its first window;
 * 
 * Runs the convolution of the series with the window.
 * 
 * @param d_QT Convolution vector on Device.
 * @param series_padded Dataset with padded zeros.
 * @param Q_padded Padded first window.
 * @param padded_size Total size of the padded data.
 * @param interval_size Window size.
 * @param mp_size Size of the matrix profile.
 */                     
void cuda_convolve(double* d_QT, 
                const double* series_padded, 
                const double* Q_padded, 
                const int padded_size,
                const int interval_size,
                const int mp_size);
