#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <memory>
#include <algorithm>

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>

using ComplexArray = std::vector<double>;

void increment_QT(const std::vector<double>& series,
                    std::vector<double>& QT,
                    const std::vector<double>& QT_old, 
                    const std::vector<double>& QT_first,
                    const int i,
                    const int final_size,
                    const int interval_size){
    
    for (int idx = 0; idx < final_size; ++idx) {
        if (idx == 0) {
            QT[idx] = QT_first[i];
        } else {
            QT[idx] = QT_old[idx - 1] 
                      - series[idx - 1] * series[i - 1] 
                      + series[idx + interval_size - 1] * series[i + interval_size - 1];
        }
    }
    
}

void distance_profile(const std::vector<double>& series, 
                        const std::vector<double>& QT,
                        const std::vector<double>& means, 
                        const std::vector<double>& stds, 
                        double* D, 
                        int* I, 
                        const int i, 
                        const int final_size, 
                        const int interval_size,
                        const int exclusion_zone_size,
                        const bool left_only){
    
    double den, Dj;
    for (int idx = 0; idx < final_size; ++idx) {

        if (abs(idx - i) >= exclusion_zone_size && (!left_only || idx<i)){

            den = interval_size*stds[i]*stds[idx];
            if (fabs(den) > 1e-9){

                double val_inside = 1.0 - (QT[idx] - interval_size * means[i] * means[idx]) / den;
                
                // Correção para imprecisões numéricas de ponto flutuante
                if (val_inside < 0.0) val_inside = 0.0;

                Dj = sqrt(2 * interval_size * val_inside);

                if (!std::isnan(Dj)) {
                    if (*D == -1.0 || Dj < *D) {
                        *D = Dj;
                        *I = idx;
                    }
                }
            }
        }
    }

}

void stomp_iteration(const std::vector<double>& series, 
                                std::vector<double>& QT,
                                const std::vector<double>& QT_old, 
                                const std::vector<double>& means, 
                                const std::vector<double>& stds, 
                                double* D, 
                                int* I, 
                                const std::vector<double>& QT_first, 
                                const int i, 
                                const int final_size, 
                                const int interval_size,
                                const int exclusion_zone_size,
                                const bool left_only){
    
    double den, Dj;
    for (int idx = 0; idx < final_size; ++idx) {

        increment_QT(series,QT,QT_old,QT_first,i,final_size,interval_size);

        if (abs(idx - i) >= exclusion_zone_size && (!left_only || idx<i)){

            den = interval_size*stds[i]*stds[idx];
            if (fabs(den) > 1e-9){

                double val_inside = 1.0 - (QT[idx] - interval_size * means[i] * means[idx]) / den;
                
                // Correção para imprecisões numéricas de ponto flutuante
                if (val_inside < 0.0) val_inside = 0.0;

                Dj = sqrt(2 * interval_size * val_inside);

                if (!std::isnan(Dj)) {
                    if (*D == -1.0 || Dj < *D) {
                        *D = Dj;
                        *I = idx;
                    }
                }
            }


        }
    }

}


// Launching STOMP
void cpu_STOMP_iterations(const std::shared_ptr<cfade::DataSet> observed_dataset,
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

    int number_of_updates = final_size-start_location;

    std::vector<double> D(number_of_updates, -1.0);
    std::vector<int> I(number_of_updates, -1);
    
    std::vector<double> QT_first(final_size);
    std::vector<double> QT_even(final_size);
    std::vector<double> QT_odd(final_size);

    
    for(int dimension = 0; dimension<observed_dataset->get_dimension();dimension++){

        std::vector<double> series_vec = (*(observed_dataset->get_data()))[dimension];
        std::vector<double> mean_vec = (*means)[dimension];
        std::vector<double> std_vec = (*stds)[dimension];


        std::fill(D.begin(),D.end(),-1.0);
        std::fill(I.begin(),I.end(),-1);

        int current_update = 0;
        bool use_odd = true;

        const double* QT_start = QT + dimension*final_size;

        std::copy(QT_start, QT_start + final_size, QT_first.begin());
        std::copy(QT_start, QT_start + final_size, QT_odd.begin());

        for(int i=0; i<start_location+1; i++){
            if(use_odd){
                increment_QT(series_vec,QT_even,QT_odd,QT_first,i,final_size,interval_size);
                use_odd = false;
            }else{
                increment_QT(series_vec,QT_odd,QT_even,QT_first,i,final_size,interval_size);
                use_odd = true;
            }

        }
        
        if(use_odd){
            distance_profile(series_vec,
                                QT_odd,
                                mean_vec,
                                std_vec,
                                &D[current_update],
                                &I[current_update],
                                start_location+current_update,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                left_only);
        }else{
            distance_profile(series_vec,
                                QT_even,
                                mean_vec,
                                std_vec,
                                &D[current_update],
                                &I[current_update],
                                start_location+current_update,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                left_only);
        }
        current_update++;

        for(; current_update<number_of_updates;current_update++){
            if (use_odd){
                stomp_iteration(series_vec,
                                QT_even,
                                QT_odd,
                                mean_vec,
                                std_vec,
                                &D[current_update],
                                &I[current_update],
                                QT_first,
                                start_location+current_update,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                left_only);
                use_odd = false;
            }else{
                stomp_iteration(series_vec,
                                QT_odd,
                                QT_even,
                                mean_vec,
                                std_vec,
                                &D[current_update],
                                &I[current_update],
                                QT_first,
                                start_location+current_update,
                                final_size,
                                interval_size,
                                exclusion_zone_size,
                                left_only);
                use_odd = true;
            }
        }

        for(int j=0;j<number_of_updates;j++){
            MP->at(dimension,j+start_location) = D[j];
            inds_MP->at(dimension,j+start_location) = I[j];
        }
    }

}



// ==========================
//  START FUNCTIONS
// ==========================


size_t get_next_power_of_two(size_t n){

    if (n <=1) return 1;
    
    n--;
    for (size_t i = 1; i < sizeof(size_t)*8; i <<= 1)
        n |= n >> i;

    return n + 1;
}

// Bit-reversal algorithm for complex array
void bit_reversal2(double *data, 
                size_t size){
    
    size_t size2 = 2*size;

    size_t j=0;
    size_t m;
    for (size_t i=0; i<size2; i+=2){
        if(j>i){
            std::swap(data[j],data[i]);
            std::swap(data[j+1],data[i+1]);
        }

        m = size;
        while(m>=2 && j>=m){
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

// Danielson-Lanczos for FFT
void inplace_danielson_lanczos_Z2Z(double *data,
                                   size_t size,
                                   int isign) {
    size_t size2 = 2 * size;
    size_t mmax = 2;
    constexpr double PI2 = 6.28318530717958647692;

    while (mmax < size2) {
        size_t istep = mmax << 1;
        double theta = -isign * ( PI2 / mmax);  // 2π/mmax

        double wtemp = std::sin(0.5 * theta);
        double wpr = -2.0 * wtemp * wtemp;
        double wpi = std::sin(theta);

        double wr = 1.0;
        double wi = 0.0;

        for (size_t m = 0; m < mmax; m += 2) {
            for (size_t i = m; i < size2; i += istep) {

                size_t j = i + mmax;
                double tempr = wr * data[j] - wi * data[j + 1];
                double tempi = wr * data[j + 1] + wi * data[j];

                data[j]     = data[i] - tempr;
                data[j + 1] = data[i + 1] - tempi;

                data[i]    += tempr;
                data[i + 1]+= tempi;

            }

            double wr_old = wr;
            wr = wr * wpr - wi * wpi + wr;
            wi = wi * wpr + wr_old * wpi + wi;
        }

        mmax = istep;
    }
}

// Inplace fft (based on numerical recipes)
void inplace_fft_Z2Z(ComplexArray& data,
                    size_t size,
                    int isign=1){

    // Getting number of cores
    // const int num_threads = std::thread::hardware_concurrency();
    // std::vector<std::thread> threads;

    // Performing bit-reversal
    bit_reversal2(data.data(),size);

    // Performing FFT
    inplace_danielson_lanczos_Z2Z(data.data(),size,isign);

}


void cpu_convolve(double *QT,
                const double* series_padded, 
                const double* Q_padded, 
                const int padded_size,
                const int interval_size,
                const int mp_size){

    // Writing complex vector
    size_t N = get_next_power_of_two(padded_size);

    ComplexArray p_series(2*N,0.0);
    ComplexArray p_query(2*N,0.0);
    ComplexArray two_inpts(2*N,0.0f);

    for(int i=0; i<padded_size;i++){
        p_series[2*i] = series_padded[i];
        p_query[2*i] = Q_padded[i];
    }
    
    // FFT
    inplace_fft_Z2Z(p_series,N,1);
    inplace_fft_Z2Z(p_query,N,1);

    // Multiply FFTs
    for (size_t i=0; i<2*N; i+=2) {

        double ar = p_series[i];
        double ai = p_series[i+1];

        double br = p_query[i];
        double bi = p_query[i+1];

        two_inpts[i]   = ar*br - ai*bi;
        two_inpts[i+1] = ar*bi + ai*br;
    }

    // IFFT
    inplace_fft_Z2Z(two_inpts,N,-1);

    double scale = 1.0 / N;
    for (size_t i = 0; i < 2*N; ++i) {
        two_inpts[i] *= scale;
    }

    // Geeting real part of multiplication
    int offset = interval_size - 1;
    for(int i=0; i<mp_size; i++){
        QT[i] = two_inpts[2*(i  + offset)];
    }

}