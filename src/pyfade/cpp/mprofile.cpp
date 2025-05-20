
#include "mprofile.h"
#include "cuda_funcs.h"

#include <iostream>
#include <numeric>
#include <cufft.h>


namespace cfade{

    Mat_Profile::Mat_Profile(const std::vector<double> series) : series(series), series_size(series.size()){}

    Mat_Profile::Mat_Profile(const std::vector<double> series, const size_t interval_size)
        : series(series), series_size(series.size()), interval_size(interval_size){}
        
    Mat_Profile::~Mat_Profile(){
        cudaFree(d_ind);
        cudaFree(d_mp);
        cudaFree(d_series);
        cudaFree(d_means);
        cudaFree(d_stds);
        cudaFree(d_QT_even);
        cudaFree(d_QT_odd);
    }

    void Mat_Profile::set_interval_size(size_t size){

        if (mp_size != 0){
            throw std::runtime_error("Cannot change interval size after calculation.");
        }

        interval_size = size;   
    }

    double Mat_Profile::return_size() const{
        return series_size;
    }

    std::vector<size_t> Mat_Profile::get_mp_ind() const{
        return mp_index;
    }

    std::vector<double> Mat_Profile::get_series() const{
        return series;
    }

    std::vector<double> Mat_Profile::get_mp() const{
        return mat_profile;
    }

    std::vector<double> Mat_Profile::get_means() const{
        return means;
    }

    std::vector<double> Mat_Profile::get_stds() const{
        return stds;
    }


    void Mat_Profile::initialize_cuda(){

        cudaMalloc((void**) &d_ind, mp_size * sizeof(size_t));
        cudaMalloc((void**) &d_mp, mp_size * sizeof(double));
        cudaMalloc((void**) &d_series, series_size * sizeof(double));
        cudaMalloc((void**) &d_means, means.size() * sizeof(double));
        cudaMalloc((void**) &d_stds, stds.size() * sizeof(double));
        cudaMalloc((void**) &d_QT_even, sizeof(double)* (series_size + interval_size-1));
        cudaMalloc((void**) &d_QT_odd, sizeof(double)* (series_size + interval_size-1));
        cudaMalloc((void**) &d_QT_first, sizeof(double)* (series_size + interval_size-1));

        cudaMemcpy(d_series, series.data(), series_size * sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy(d_means, means.data(), means.size() * sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy(d_stds, stds.data(), stds.size() * sizeof(double), cudaMemcpyHostToDevice );

        cudaMemset(d_mp, 0, mp_size* sizeof(double));
        cudaMemset(d_ind, 0, mp_size* sizeof(size_t));

    }

    void Mat_Profile::get_data_from_cuda(){

        cudaMemcpy(mp_index.data(), d_ind, mp_size * sizeof(size_t), cudaMemcpyDeviceToHost );
        cudaMemcpy(mat_profile.data(), d_mp, mp_size * sizeof(double), cudaMemcpyDeviceToHost );


    }

    void Mat_Profile::get_first_product(){

        // Getting flipped first interval
        std::vector<double> Q(series_size,0.0f);
        for (int i=0; i<interval_size; i++){
            Q[i] = series[series_size-1-i];
        }

        // Padding arrays (also increasing size so everything is ok)
        size_t padded_size = series_size + interval_size-1;
        std::vector<double> Q_padded(padded_size,0.0f), series_padded(padded_size,0.0f);
        std::copy(series.begin(), series.end(), series_padded.begin());
        std::copy(Q.begin(), Q.end(), Q_padded.begin());

        // Passing stuff to device
        double *d_Q, *d_series_padded;
        cufftDoubleComplex *d_fft_series, *d_fft_Q;

        cudaMalloc(&d_series_padded, sizeof(double)*padded_size);
        cudaMalloc(&d_Q, sizeof(double)*padded_size);
    
        size_t fft_size = padded_size/2+1;
        cudaMalloc(&d_fft_series, sizeof(cufftDoubleComplex) * fft_size); /////////
        cudaMalloc(&d_fft_Q, sizeof(cufftDoubleComplex) * fft_size); /////////////

        cudaMemcpy(d_series_padded,series_padded.data(), sizeof(double)*padded_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q,Q_padded.data(), sizeof(double)*padded_size, cudaMemcpyHostToDevice);


        // Running FFT
        cufftHandle plan_fft;
        cufftPlan1d(&plan_fft, padded_size, CUFFT_D2Z, 1);
        cufftExecD2Z(plan_fft, d_series_padded, d_fft_series);
        cufftExecD2Z(plan_fft, d_Q, d_fft_Q);

        // Multiplication
        launch_multiply_complex(d_fft_series, d_fft_Q, fft_size);
        cudaDeviceSynchronize();

        // Inverse FFT
        cufftHandle plan_ifft;
        cufftPlan1d(&plan_ifft, padded_size, CUFFT_Z2D, 1);
        cufftExecZ2D(plan_ifft, d_fft_series, d_QT_even);

        cufftDestroy(plan_fft);
        cufftDestroy(plan_ifft);
        cudaFree(d_series_padded);
        cudaFree(d_fft_series);
        cudaFree(d_fft_Q);
    }

    void Mat_Profile::run_batch(){

        // Checking sizes
        if(interval_size == 0){
            throw std::runtime_error("Must set interval size first");
        }
        size_checker(series_size);

        // Initializing means and standard deviations
        mp_size = series_size-interval_size+1;
        means = calculate_means(series, interval_size);
        stds = calculate_stds(series, interval_size,means);

        // Performing first product
        initialize_cuda();
        get_first_product();


        // Starting MP and index
        launch_distance_profile(d_series,
                                d_QT_even,
                                d_QT_first,
                                d_means, 
                                d_stds, 
                                d_mp, 
                                d_ind, 
                                0, 
                                mp_size, 
                                interval_size,
                                exclusion_zone_size,
                                skip_start,
                                left_only);
        current_QT_location=1;


        // Running iterations
        launch_STOMP_iterations(d_series, 
                        d_QT_even,
                        d_QT_odd, 
                        d_means, 
                        d_stds, 
                        d_mp, 
                        d_ind, 
                        d_QT_first, 
                        current_QT_location, 
                        mp_size, 
                        interval_size,
                        exclusion_zone_size,
                        skip_start,
                        left_only);


        // Getting CUDA data
        mat_profile = std::vector<double>(mp_size);
        // QT          = std::vector<double>(mp_size);
        mp_index    = std::vector<size_t>(mp_size);
        get_data_from_cuda();


        std::cout << current_QT_location << std::endl;



    }



    // Getting the mean value of each subsequence
    std::vector<double> calculate_means(const std::vector<double>& series, size_t interval_size){

        // Checking sizes
        size_checker(interval_size);
        size_checker(series.size());

        
        // Getting cumsum of values
        std::vector<double> cumsum(series.size()+1);
        cumsum[0] = 0;
        std::partial_sum(series.begin(),series.end(),cumsum.begin()+1);

        // Getting the mean values
        size_t final_size = series.size()-interval_size+1;
        std::vector<double> means(final_size);
        for(size_t i=0; i<final_size; i++){
            means[i] = (cumsum.at(i+interval_size) - cumsum.at(i))/interval_size;
        }

        return means;

    }

    std::vector<double> calculate_stds(const std::vector<double>& series, size_t interval_size, const std::vector<double>& means){

        // Checking sizes
        size_checker(interval_size);
        size_checker(series.size());
        size_checker(means.size());

        
        // Getting cumsum of values
        std::vector<double> sq_series(series.size());
        for (size_t i=0; i<series.size();i++){
            sq_series[i] = series.at(i)*series.at(i);
        }

        std::vector<double> cumsum(series.size()+1);
        cumsum[0] = 0;
        std::partial_sum(sq_series.begin(),sq_series.end(),cumsum.begin()+1);

        // Getting the mean values
        size_t final_size = series.size()-interval_size+1;
        std::vector<double> stds(final_size);
        for(size_t i=0; i<final_size; i++){
            stds[i] = ( (cumsum.at(i+interval_size) - cumsum.at(i))/interval_size - means.at(i)*means.at(i) );
        }

        return stds;

    }


    void size_checker(size_t size){
        if (size<=0){

            throw std::runtime_error("Size of array must be positive");
        }
    }
    

} // namespace cfade