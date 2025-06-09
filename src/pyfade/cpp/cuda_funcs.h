#pragma once

#include <cpp/dataset.h>
#include <cstddef>
#include <cufft.h>
#include <memory>

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

void cuda_convolve(double* d_QT, 
                const double* series_padded, 
                const double* Q_padded, 
                const int padded_size,
                const int interval_size,
                const int mp_size);
