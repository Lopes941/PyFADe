#include "cuda_funcs.h"
#include <iostream>
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


void launch_multiply_complex(cufftDoubleComplex* d_fft_series, 
                                const cufftDoubleComplex* d_fft_kernel, 
                                int fft_size){

    const int threads = 256;
    const int blocks = (fft_size + threads - 1) / threads;
    multiply_complex<<<blocks, threads>>>(d_fft_series, d_fft_kernel, fft_size);

}

void launch_iff_normalization(double* data, 
                            int size){

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    normalize_ifft<<<blocks, threads>>>(data, size);

}


// Launching STOMP
void launch_STOMP_iterations(const double* series, 
                        double* QT,
                        const double* means, 
                        const double* stds, 
                        double* MP, 
                        int* inds_MP,
                        int &current_QT_location, 
                        const int final_size, 
                        const int interval_size,
                        const int exclusion_zone_size,
                        const bool left_only){

    const int threads = 256;
    const int blocks = (final_size + threads - 1) / threads;
    double* QT_first, *QT_2;
    double *D;
    int *I;

    int start_location = current_QT_location;
    int number_of_updates = final_size-current_QT_location;
    int current_update = 0;

    cudaMalloc(&D, sizeof(double)*number_of_updates);
    cudaMalloc(&I, sizeof(int)*number_of_updates);

    fill_double<<<blocks,threads>>>(D,-1,number_of_updates);
    fill_int<<<blocks,threads>>>(I,-1,number_of_updates);
    

    // Getting first QT if iteration 0
    if (current_QT_location==0){
                
        distance_profile<<<blocks, threads>>>(series,
                                            QT,
                                            means,
                                            stds,
                                            D+current_update,
                                            I+current_update,
                                            current_QT_location,
                                            final_size,
                                            interval_size,
                                            exclusion_zone_size,
                                            left_only);
        current_QT_location++;
        current_update++;
    }

    cudaMalloc((void**) &QT_first, sizeof(double)* final_size);
    cudaMalloc((void**) &QT_2, sizeof(double)* final_size);

    cudaMemcpy(QT_first, QT, final_size * sizeof(double), cudaMemcpyDeviceToDevice );
    cudaMemcpy(QT_2, QT, final_size * sizeof(double), cudaMemcpyDeviceToDevice );

    for(current_QT_location; current_QT_location<final_size;current_QT_location++){
        if (current_QT_location%2){
            stomp_iteration<<<blocks, threads>>>(series,
                                QT_2,
                                QT,
                                means,
                                stds,
                                D+current_update,
                                I+current_update,
                                QT_first,
                                current_QT_location,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                left_only);
        }else{
            stomp_iteration<<<blocks, threads>>>(series,
                                QT,
                                QT_2,
                                means,
                                stds,
                                D+current_update,
                                I+current_update,
                                QT_first,
                                current_QT_location,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                left_only);
        }
        current_update++;
    }

    cudaMemcpy(MP+start_location, D, sizeof(double)*number_of_updates, cudaMemcpyDeviceToHost);
    cudaMemcpy(inds_MP+start_location, I, sizeof(int)*number_of_updates, cudaMemcpyDeviceToHost);



    cudaFree(QT_first);
    cudaFree(QT_2);
    cudaFree(D);
    cudaFree(I);


}