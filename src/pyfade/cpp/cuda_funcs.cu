#include "cuda_funcs.h"

    // ==========================
    //  Kernels
    // ==========================

    
__global__ void stomp_iteration(const double* series, 
                                double* QT,
                                const double* QT_old, 
                                const double* means, 
                                const double* stds, 
                                double* MP, 
                                size_t* inds_MP, 
                                const double* QT_first, 
                                const size_t i, 
                                const size_t final_size, 
                                const size_t interval_size,
                                const size_t exclusion_zone_size,
                                const size_t initial_size,
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

        if (idx >=initial_size && (idx>=i+exclusion_zone_size || (idx<i-exclusion_zone_size && !left_only))){

            den = interval_size*stds[i]*stds[idx];
            if (fabs(den) < 1e-9){
                den = 1e-9;
            }
            Dj = sqrt(2 * interval_size * ( 1- (QT[idx] - interval_size*means[i]*means[idx] )/den ));

            if (MP[idx] > Dj){
                MP[idx] = Dj;
                inds_MP[idx] = i;
            }

        }
    }

}

__global__ void distance_profile(const double* series, 
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
                                    const bool left_only){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double den;


    if (idx < final_size) {

        if (idx >=initial_size && (idx>=i+exclusion_zone_size || (idx<i-exclusion_zone_size && !left_only))){

            den = interval_size*stds[i]*stds[idx];
            if (fabs(den) < 1e-9){
                den = 1e-9;
            }

            d_QT_first[idx] = QT[idx];
            MP[idx]         = sqrt(2 * interval_size * ( 1- (QT[idx] - interval_size*means[i]*means[idx] )/den ));
            inds_MP[idx]    = i;

        }
    }

}

__global__  void multiply_complex(cufftDoubleComplex* A, 
                                const cufftDoubleComplex* B, 
                                size_t N){

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double a = A[i].x, b = A[i].y;
        double c = B[i].x, d = B[i].y;
        A[i].x = a * c - b * d;
        A[i].y = a * d + b * c;
    }
}





// ==========================
//  LAUNCH FUNCTIONS
// ==========================


void launch_multiply_complex(cufftDoubleComplex* d_fft_series, 
                                const cufftDoubleComplex* d_fft_kernel, 
                                size_t fft_size){

    const int threads = 256;
    const int blocks = (fft_size + threads - 1) / threads;
    multiply_complex<<<blocks, threads>>>(d_fft_series, d_fft_kernel, fft_size);
}


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
                                const bool left_only){

    const int threads = 256;
    const int blocks = (final_size + threads - 1) / threads;

    distance_profile<<<blocks, threads>>>(series,
                                            QT,
                                            d_QT_first,
                                            means,
                                            stds,
                                            MP,
                                            inds_MP,
                                            i,
                                            final_size,
                                            interval_size,
                                            exclusion_zone_size,
                                            initial_size,
                                            left_only);
}

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
                        const bool left_only){

    const int threads = 256;
    const int blocks = (final_size + threads - 1) / threads;

    for(i; i<final_size;i++){
        if (i%2){
            stomp_iteration<<<blocks, threads>>>(series,
                                QT_odd,
                                QT_even,
                                means,
                                stds,
                                MP,
                                inds_MP,
                                QT_first,
                                i,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                initial_size,
                                left_only);
        }else{
            stomp_iteration<<<blocks, threads>>>(series,
                                QT_even,
                                QT_odd,
                                means,
                                stds,
                                MP,
                                inds_MP,
                                QT_first,
                                i,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                initial_size,
                                left_only);
        }
    }

    for(int j=0; j<initial_size; j++){
        MP[j] = 0;
    }


}