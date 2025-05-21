
#include "mprofile.h"
#include "cuda_funcs.h"

#include <iostream>
#include <numeric>
#include <cufft.h>


namespace cfade{

    // Default constructor
    Mat_Profile::Mat_Profile() {}

    // Constructor with given series
    Mat_Profile::Mat_Profile(const std::vector<double> series_) : series(series_), append_size(series_.size()){}

    // Constructor with given series and interval size
    Mat_Profile::Mat_Profile(const std::vector<double> series_, const int interval_size_)
        : series(series_), interval_size(interval_size_), append_size(series_.size()){
            exclusion_zone_size = (int) std::round(exclusion_zone_ratio*interval_size_);
        }

    // Default destructor
    Mat_Profile::~Mat_Profile(){
        if (d_series) cudaFree(d_series);
        if (d_means) cudaFree(d_means);
        if (d_stds) cudaFree(d_stds);
        if (d_QT) cudaFree(d_QT);

        d_series = nullptr;
        d_means = nullptr;
        d_stds = nullptr;
        d_QT = nullptr;
    }

    // Setter for interval size
    void Mat_Profile::set_interval_size(int size){

        if (mp_size != 0){
            throw std::runtime_error("Cannot change interval size after calculation.");
        }

        interval_size = size;   
        exclusion_zone_size = (int) std::round(exclusion_zone_ratio*interval_size);
    }

    // Setter for exclusion zone ratio (ratio between exclusion zone size and interval size)
    void Mat_Profile::set_exclusion_ratio(const float ratio){
        exclusion_zone_ratio = ratio;
        exclusion_zone_size = (int) std::round(ratio*interval_size);
    }

    // Setter for initial data skip
    void Mat_Profile::set_start_ignore(const int skip){
        skip_start = skip;
    }

    // Setter for left only flag
    void Mat_Profile::set_left_only(const bool left_only_){
        left_only = left_only_;
    }

    //  Getter for size of series
    double Mat_Profile::get_size() const{
        return series.size();
    }

    // Getter for matrix profile index data
    std::vector<int> Mat_Profile::get_mp_ind() const{
        return mp_index;
    }

    // Getter for series  data
    std::vector<double> Mat_Profile::get_series() const{
        return series;
    }

    // Getter for matrix profile data
    std::vector<double> Mat_Profile::get_mp() const{
        return mat_profile;
    }

    // Getter for means data
    std::vector<double> Mat_Profile::get_means() const{
        return means;
    }

    // Getter for standard deviation data
    std::vector<double> Mat_Profile::get_stds() const{
        return stds;
    }

    // Getter for current convolution
    std::vector<double> Mat_Profile::get_QT() const{
        return QT;
    }

    // Initialization of CUDA pointers
    void Mat_Profile::initialize_cuda(){

        started_runtime = true;
        // cudaMalloc((void**) &d_ind, mp_size * sizeof(int));
        // cudaMalloc((void**) &d_mp, mp_size * sizeof(double));
        cudaMalloc((void**) &d_series, series.size() * sizeof(double));
        cudaMalloc((void**) &d_means, means.size() * sizeof(double));
        cudaMalloc((void**) &d_stds, stds.size() * sizeof(double));
        cudaMalloc((void**) &d_QT, sizeof(double)* mp_size);
        

        cudaMemcpy(d_series, series.data(), series.size() * sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy(d_means, means.data(), means.size() * sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy(d_stds, stds.data(), stds.size() * sizeof(double), cudaMemcpyHostToDevice );

        if(mat_profile.size()>0){
            // cudaMemcpy(d_mp, mat_profile.data(), mp_size * sizeof(double), cudaMemcpyHostToDevice );
            cudaMemcpy(d_QT, QT.data(), mp_size * sizeof(double), cudaMemcpyHostToDevice );
            // cudaMemcpy(d_ind, mp_index.data(), mp_size * sizeof(int), cudaMemcpyHostToDevice );
        }else{
            // cudaMemset(d_mp, 0, mp_size* sizeof(double));
            cudaMemset(d_QT, 0, mp_size* sizeof(double));
            // cudaMemset(d_ind, -1, mp_size* sizeof(int));
        }

        
        

    }

    // Passing CUDA pointers to host and freeing them
    void Mat_Profile::release_from_cuda(){

        started_runtime = false;

        QT = std::vector<double>(mp_size);

        cudaMemcpy(QT.data(), d_QT, mp_size * sizeof(double), cudaMemcpyDeviceToHost );


        if (d_series) cudaFree(d_series);
        if (d_means) cudaFree(d_means);
        if (d_stds) cudaFree(d_stds);
        if (d_QT) cudaFree(d_QT);

        d_series = nullptr;
        d_means = nullptr;
        d_stds = nullptr;
        d_QT = nullptr;
    }

    // Calculation of first dot product via CuFFT
    void Mat_Profile::get_first_product(int start){

        // Getting flipped first interval padded
        int padded_size = series.size() + interval_size-1;
        std::vector<double> Q_padded(padded_size,0.0f);
        for (int i=0; i<interval_size; i++){
            Q_padded[i] = series[interval_size-1-i+start];
        }

        // Padding arrays (also increasing size so everything is ok)
        std::vector<double> series_padded(padded_size,0.0f);
        std::copy(series.begin(), series.end(), series_padded.begin());

        // Passing stuff to device
        double *d_Q, *d_series_padded, *d_QT_padded;
        cufftDoubleComplex *d_fft_series, *d_fft_Q;

        cudaMalloc(&d_series_padded, sizeof(double)*padded_size);
        cudaMalloc(&d_Q, sizeof(double)*padded_size);
        cudaMalloc(&d_QT_padded, sizeof(double)*padded_size);
    
        int fft_size = padded_size/2+1;
        cudaMalloc(&d_fft_series, sizeof(cufftDoubleComplex) * fft_size);
        cudaMalloc(&d_fft_Q, sizeof(cufftDoubleComplex) * fft_size);

        cudaMemcpy(d_series_padded,series_padded.data(), sizeof(double)*padded_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Q,Q_padded.data(), sizeof(double)*padded_size, cudaMemcpyHostToDevice);


        // Running FFT
        cufftHandle plan_fft_series, plan_fft_Q;
        cufftPlan1d(&plan_fft_series, padded_size, CUFFT_D2Z, 1);
        cufftPlan1d(&plan_fft_Q, padded_size, CUFFT_D2Z, 1);
        cufftExecD2Z(plan_fft_series, d_series_padded, d_fft_series);
        cufftExecD2Z(plan_fft_Q, d_Q, d_fft_Q);

        // Multiplication
        launch_multiply_complex(d_fft_series, d_fft_Q, fft_size);

        // Inverse FFT
        cufftHandle plan_ifft;
        cufftPlan1d(&plan_ifft, padded_size, CUFFT_Z2D, 1);
        cufftExecZ2D(plan_ifft, d_fft_series, d_QT_padded);
        launch_iff_normalization(d_QT_padded, padded_size);
        cudaMemcpy(d_QT, d_QT_padded+interval_size-1, mp_size * sizeof(double), cudaMemcpyDeviceToDevice );


        cufftDestroy(plan_fft_series);
        cufftDestroy(plan_fft_Q);
        cufftDestroy(plan_ifft);

        cudaFree(d_series_padded);
        cudaFree(d_fft_series);
        cudaFree(d_fft_Q);
        cudaFree(d_QT_padded);
    }

    // Starting runtime operations
    void Mat_Profile::init_runtime(){
        initialize_cuda();
    }

    // Stopping runtime operation
    void Mat_Profile::stop_runtime(){
        release_from_cuda();
    }



    void Mat_Profile::run_batch(){

        // Checking sizes
        if(interval_size == 0){
            throw std::runtime_error("Must set interval size first");
        }

        if(append_size == 0){
            std::cout << "WARNING: Must include new data" << std::endl;
        }

        // Initializing mp
        if (series.size()<interval_size) return;
        if (mp_size == 0){
            reserve_mp(series.size()-interval_size+1);
        }else{
            reserve_mp(append_size);
        }
        
        
        // Initializing means and standard deviations
        means = calculate_means(series, interval_size);
        stds = calculate_stds(series, interval_size,means);

        // Performing first product
        bool started_here=false;
        if(!started_runtime){
            initialize_cuda();
            started_here=true;
        }
        
        get_first_product(current_QT_location);


        // Running iterations
        launch_STOMP_iterations(d_series, 
                        d_QT,
                        d_means, 
                        d_stds, 
                        mat_profile.data(), 
                        mp_index.data(), 
                        current_QT_location, 
                        mp_size, 
                        interval_size,
                        exclusion_zone_size,
                        left_only);

        // Removing start
        if(mp_size>skip_start){
            for(int i=0; i<skip_start; i++){
                mat_profile[i] = 0;
            }
        }


        // Getting CUDA data
        if(started_here){
            release_from_cuda();
        }else{
            QT = std::vector<double>(mp_size);
            cudaMemcpy(QT.data(), d_QT, mp_size * sizeof(double), cudaMemcpyDeviceToHost );
        }
        


    }


    
    

    void Mat_Profile::append_data(std::vector<double> appended_data){
        append_size += appended_data.size();
        series.reserve(append_size+series.size());
        series.insert(series.end(),appended_data.begin(),appended_data.end());
    }

    void Mat_Profile::reserve_mp(size_t incremented_size){
        mp_index.reserve(incremented_size+mp_size);
        mat_profile.reserve(incremented_size+mp_size);

        mp_index.insert(mp_index.end(),incremented_size,-1);
        mat_profile.insert(mat_profile.end(),incremented_size,1e9);

        mp_size += incremented_size;
    }

    // Getting the mean value of each subsequence
    std::vector<double> calculate_means(const std::vector<double>& series, int interval_size){

        // Getting cumsum of values
        std::vector<double> cumsum(series.size()+1);
        cumsum[0] = 0;
        std::partial_sum(series.begin(),series.end(),cumsum.begin()+1);

        // Getting the mean values
        int final_size = series.size()-interval_size+1;
        std::vector<double> means(final_size);
        for(int i=0; i<final_size; i++){
            means[i] = (cumsum.at(i+interval_size) - cumsum.at(i))/interval_size;
        }

        return means;

    }

    std::vector<double> calculate_stds(const std::vector<double>& series, int interval_size, const std::vector<double>& means){


        // Getting cumsum of values
        std::vector<double> sq_series(series.size());
        for (int i=0; i<series.size();i++){
            sq_series[i] = series.at(i)*series.at(i);
        }

        std::vector<double> cumsum(series.size()+1);
        cumsum[0] = 0;
        std::partial_sum(sq_series.begin(),sq_series.end(),cumsum.begin()+1);

        // Getting the mean values
        int final_size = series.size()-interval_size+1;
        std::vector<double> stds(final_size);
        for(int i=0; i<final_size; i++){
            stds[i] = std::sqrt( (cumsum.at(i+interval_size) - cumsum.at(i))/interval_size - means.at(i)*means.at(i) );
        }

        return stds;

    }

} // namespace cfade