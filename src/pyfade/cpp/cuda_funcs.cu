
#include <cpp/dataset.h>
#include <cpp/matrix_profile.h>
#include "cuda_funcs.h"

#include <iostream>
#include <memory>

    // ==========================
    //  Kernels
    // ==========================

__device__ void atomicMin_double_with_index_nan_safe(double* addr_val, int* addr_idx, double val, int idx) {
    if (isnan(val)) return;

    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(addr_val);
    unsigned long long old = *address_as_ull, assumed;
    double old_val;

    do {
        assumed = old;
        old_val = __longlong_as_double(assumed);

        if (old_val!=-1 && old_val <= val) return;

        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    // Only set index if value changed
    atomicExch(addr_idx, idx);
}

__global__ void stomp_iteration(const double* series, 
                                double* QT,
                                const double* QT_old, 
                                const double* means, 
                                const double* stds, 
                                double* MP, 
                                int* inds_MP, 
                                const double* QT_first, 
                                const int i, 
                                const int final_size, 
                                const int interval_size,
                                const int exclusion_zone_size,
                                const bool left_only){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double Dj, den;


    if (idx < final_size) {

        if (idx == 0){
            QT[idx] = QT_first[i];
        }else{
            QT[idx] =   QT_old[idx-1] 
                        - series[idx-1]*series[i-1] 
                        + series[idx+interval_size-1]*series[i+interval_size-1];
        }

        if (abs(idx - i) >= exclusion_zone_size && (!left_only || idx<i)){

            den = interval_size*stds[i]*stds[idx];
            if (fabs(den) > 1e-9){
                Dj = sqrt(2 * interval_size * ( 1- (QT[idx] - interval_size*means[i]*means[idx] )/den ));
                atomicMin_double_with_index_nan_safe(MP,inds_MP,Dj,idx);
            }
            

        }
    }

}

__global__ void distance_profile(const double* series, 
                                    const double* QT,
                                    const double* means, 
                                    const double* stds, 
                                    double* MP, 
                                    int* inds_MP, 
                                    const int i, 
                                    const int final_size, 
                                    const int interval_size,
                                    const int exclusion_zone_size,
                                    const bool left_only){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double den, Dj;


    if (idx < final_size) {

        if (abs(idx - i) >= exclusion_zone_size && (!left_only || idx<i)){

            den = interval_size*stds[i]*stds[idx];
            if (fabs(den) > 1e-9){
                Dj = sqrt(2 * interval_size * ( 1- (QT[idx] - interval_size*means[i]*means[idx] )/den ));
                atomicMin_double_with_index_nan_safe(MP,inds_MP,Dj,idx);
            }

        }
    }

}

__global__  void multiply_complex(cufftDoubleComplex* A, 
                                const cufftDoubleComplex* B, 
                                int N){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double a = A[idx].x, b = A[idx].y;
        double c = B[idx].x, d = B[idx].y;
        A[idx].x = a * c - b * d;
        A[idx].y = a * d + b * c;
    }
}

__global__  void normalize_ifft(double* data, 
                                int size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= size;
    }
}

__global__ void fill_double(double* d_arr, double var, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_arr[idx] = var;
}

__global__ void fill_int(int* d_arr, int var, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        d_arr[idx] = var;
}

// ==========================
//  LAUNCH FUNCTIONS
// ==========================

// Number of blocks
void choose_cuda_parameters(int total_size,
                            int &num_blocks,
                            int &num_threads){
    
    // Getting GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);

    // Choosing number of threads
    num_threads = 256;
    if (num_threads > prop.maxThreadsPerBlock)
        num_threads = prop.maxThreadsPerBlock;

    // Computing number of blocks
    num_blocks = static_cast<int>((total_size+num_threads-1)/num_threads);

}

// Launching STOMP
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
                        const bool left_only){

    int threads;
    int blocks;

    choose_cuda_parameters(final_size,blocks,threads);
    double* QT_first, *QT_even, *QT_odd;
    double *d_mean, *d_series, *d_stds;
    double *D;
    int *I;

    int number_of_updates = final_size-start_location;


    cudaMalloc(&D, sizeof(double)*number_of_updates);
    cudaMalloc(&I, sizeof(int)*number_of_updates);
    cudaMalloc((void**) &QT_first, sizeof(double)* final_size);
    cudaMalloc((void**) &QT_even, sizeof(double)* final_size);
    cudaMalloc((void**) &QT_odd, sizeof(double)* final_size);
    cudaMalloc((void**) &d_mean, sizeof(double)* final_size);
    cudaMalloc((void**) &d_series, sizeof(double)* observed_dataset->get_length());
    cudaMalloc((void**) &d_stds, sizeof(double)* final_size);

    for(int dimension = 0; dimension<observed_dataset->get_dimension();dimension++){

        std::vector<double> series_vec = (*(observed_dataset->get_data()))[dimension];
        std::vector<double> mean_vec = (*means)[dimension];
        std::vector<double> std_vec = (*stds)[dimension];

        cudaMemcpy(d_mean, mean_vec.data(), final_size * sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy(d_stds, std_vec.data(), final_size * sizeof(double), cudaMemcpyHostToDevice );
        cudaMemcpy(d_series, series_vec.data(), observed_dataset->get_length() * sizeof(double), cudaMemcpyHostToDevice );

        fill_double<<<blocks,threads>>>(D,-1,number_of_updates);
        fill_int<<<blocks,threads>>>(I,-1,number_of_updates);

        int current_update = 0;

        distance_profile<<<blocks, threads>>>(d_series,
                                            QT,
                                            d_mean,
                                            d_stds,
                                            D+current_update,
                                            I+current_update,
                                            start_location+current_update,
                                            final_size,
                                            interval_size,
                                            exclusion_zone_size,
                                            left_only);
        current_update++;

        cudaMemcpy(QT_first, QT+dimension*final_size, final_size * sizeof(double), cudaMemcpyDeviceToDevice );
        cudaMemcpy(QT_odd, QT+dimension*final_size,final_size * sizeof(double), cudaMemcpyDeviceToDevice );


        for(; current_update<number_of_updates;current_update++){
            if (current_update%2){
                stomp_iteration<<<blocks, threads>>>(d_series,
                                    QT_even,
                                    QT_odd,
                                    d_mean,
                                    d_stds,
                                    D+current_update,
                                    I+current_update,
                                    QT_first,
                                    start_location+current_update,
                                    final_size,
                                    interval_size,
                                    exclusion_zone_size,
                                    left_only);
            }else{
                stomp_iteration<<<blocks, threads>>>(d_series,
                                    QT_odd,
                                    QT_even,
                                    d_mean,
                                    d_stds,
                                    D+current_update,
                                    I+current_update,
                                    QT_first,
                                    start_location+current_update,
                                    final_size,
                                    interval_size,
                                    exclusion_zone_size,
                                    left_only);
            }
        }

        std::vector<double> matrix_profile_vec(number_of_updates);
        std::vector<int> matrix_index_vec(number_of_updates);

        cudaMemcpy(matrix_profile_vec.data(), D, sizeof(double)*number_of_updates, cudaMemcpyDeviceToHost);
        cudaMemcpy(matrix_index_vec.data(), I, sizeof(int)*number_of_updates, cudaMemcpyDeviceToHost);

        for(int j=0;j<number_of_updates;j++){
            MP->at(dimension,j+start_location) = matrix_profile_vec[j];
            inds_MP->at(dimension,j+start_location) = matrix_index_vec[j];
        }
    }

    cudaFree(d_mean);
    cudaFree(d_stds);
    cudaFree(d_series);
    cudaFree(QT_first);
    cudaFree(QT_odd);
    cudaFree(QT_even);
    cudaFree(D);
    cudaFree(I);

}

void cuda_convolve(double* d_QT, 
                const double* series_padded, 
                const double* Q_padded, 
                const int padded_size,
                const int interval_size,
                const int mp_size){


    int threads;

    // Passing stuff to device
    double *d_Q, *d_series_padded, *d_QT_padded;
    cufftDoubleComplex *d_fft_series, *d_fft_Q;

    cudaMalloc(&d_series_padded, sizeof(double)*padded_size);
    cudaMalloc(&d_Q, sizeof(double)*padded_size);
    cudaMalloc(&d_QT_padded, sizeof(double)*padded_size);

    int fft_size = padded_size/2+1;
    cudaMalloc(&d_fft_series, sizeof(cufftDoubleComplex) * fft_size);
    cudaMalloc(&d_fft_Q, sizeof(cufftDoubleComplex) * fft_size);

    cudaMemcpy(d_series_padded,series_padded, sizeof(double)*padded_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q,Q_padded, sizeof(double)*padded_size, cudaMemcpyHostToDevice);


    // Running FFT
    cufftHandle plan_fft_series, plan_fft_Q;
    cufftPlan1d(&plan_fft_series, padded_size, CUFFT_D2Z, 1);
    cufftPlan1d(&plan_fft_Q, padded_size, CUFFT_D2Z, 1);
    cufftExecD2Z(plan_fft_series, d_series_padded, d_fft_series);
    cufftExecD2Z(plan_fft_Q, d_Q, d_fft_Q);

    // Multiplication
    
    int blocks_mult;
    choose_cuda_parameters(fft_size,blocks_mult,threads);
    multiply_complex<<<blocks_mult, threads>>>(d_fft_series, d_fft_Q, fft_size);

    // Inverse FFT
    int blocks_normalization;
    choose_cuda_parameters(fft_size,blocks_normalization,threads);
    cufftHandle plan_ifft;
    cufftPlan1d(&plan_ifft, padded_size, CUFFT_Z2D, 1);
    cufftExecZ2D(plan_ifft, d_fft_series, d_QT_padded);
    normalize_ifft<<<blocks_normalization, threads>>>(d_QT_padded, padded_size);
    cudaMemcpy(d_QT, d_QT_padded+interval_size-1, mp_size * sizeof(double), cudaMemcpyDeviceToDevice );


    cufftDestroy(plan_fft_series);
    cufftDestroy(plan_fft_Q);
    cufftDestroy(plan_ifft);

    cudaFree(d_series_padded);
    cudaFree(d_fft_series);
    cudaFree(d_fft_Q);
    cudaFree(d_QT_padded);

}


