
#include <cpp/window.h>
#include <cpp/dataset.h>
#include <cpp/matrix_profile.h>


// #ifdef USE_CUDA
    #include "cuda_funcs.h"
    #include <cufft.h>
    #include <cuda_runtime.h>
// #endif


namespace cfade
{
    MatrixProfile::MatrixProfile(const std::shared_ptr<DataSet> observed_dataset,
                                const std::vector<FeatureGroupInterface> dependencies):
    matrix_profile(std::make_shared<VectorGroup<double>>(observed_dataset->get_dimension())),
    index(std::make_shared<VectorGroup<int>>(observed_dataset->get_dimension())),
    means(dependencies[0].get_means()),
    stds(dependencies[0].get_stds())
    {}

    void MatrixProfile::update(const std::shared_ptr<DataSet> observed_dataset,
                        int window_size){


        QTDataInterface QT;
        int start_loc = matrix_profile->cols;
        int final_size = observed_dataset->get_size() - window_size + 1;

        matrix_profile->increase_cols(final_size-start_loc);
        index->increase_cols(final_size-start_loc);
        if (use_cuda){
            QT = QTDataCUDA(observed_dataset, means, stds, window_size, start_loc);
        }else{
            // TODO
        }

        if(use_cuda){
            cuda_STOMP_iterations(observed_dataset, 
                            means,
                            stds, 
                            QT.d_QT, 
                            matrix_profile, 
                            index, 
                            start_loc, 
                            final_size, 
                            window_size,
                            exclusion_zone_size,
                            left_only);
        }

        // TODO
        
        
    }
    


    // #ifdef USE_CUDA
        QTDataCUDA::QTDataCUDA(const std::shared_ptr<DataSet> observed_dataset,
                                const std::shared_ptr<VectorGroup<double>> means,
                                const std::shared_ptr<VectorGroup<double>> stds,
                                int window_size,
                                int start_loc): d_QT(nullptr){

            cudaMalloc((void**) &d_QT, sizeof(double) * means->size());

            // Getting flipped first interval padded
            int padded_size = observed_dataset->get_size() + window_size-1;
            int mp_size = means->cols;
            std::vector<double> Q_padded(padded_size,0.0f);
            std::vector<double> series_padded(padded_size,0.0f);

            for(int dimension=0; dimension<observed_dataset->get_dimension(); dimension++){

                for (int i=0; i<window_size; i++){
                    Q_padded[i] = observed_dataset->at(dimension,window_size-1-i + start_loc);
                }
                for(int i=0;i<observed_dataset->get_size(); i++){
                    series_padded[i] = observed_dataset->at(dimension,i);
                }

                cuda_convolve(d_QT+mp_size*dimension, series_padded.data(), Q_padded.data(), padded_size, window_size,mp_size);

            }
        }

        QTDataCUDA::~QTDataCUDA(){

            if(d_QT) cudaFree(d_QT);

        }

        

    // #endif

} // namespace cfade