#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

int get_next_power_of_two(int n){
    if (n <=0) return 1;
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Bit-reversal algorithm for complex array
void bit_reversal2(double *data, 
                int size){
    
    int size2 = 2*size;

    int j=0;
    int m;
    for (int i=0; i<size2; i+=2){
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
                                   int size,
                                   int isign) {
    int size2 = 2 * size;
    int mmax = 2;

    while (mmax < size2) {
        int istep = mmax << 1;
        double theta = isign * (6.28318530717958647692 / mmax);  // 2π/mmax

        for (int m = 0; m < mmax; m += 2) {
            double angle = (m / 2) * theta;
            double wr = std::cos(angle);
            double wi = std::sin(angle);

            for (int i = m; i < size2; i += istep) {
                int j = i + mmax;
                double tempr = wr * data[j] - wi * data[j + 1];
                double tempi = wr * data[j + 1] + wi * data[j];

                data[j]     = data[i] - tempr;
                data[j + 1] = data[i + 1] - tempi;
                data[i]    += tempr;
                data[i + 1]+= tempi;
            }
        }

        mmax = istep;
    }
}

// Inplace fft (based on numerical recipes)
void inplace_fft_Z2Z(double *data,
                    size_t size,
                    int isign=1){

    // Getting number of cores
    // const int num_threads = std::thread::hardware_concurrency();
    // std::vector<std::thread> threads;

    // Performing bit-reversal
    bit_reversal2(data,size);

    // Performing FFT
    inplace_danielson_lanczos_Z2Z(data,size,isign);

}

// Inplace fft (based on numerical recipes)
void inplace_mult_fft(double *data, size_t size) {
    int size2 = 2 * size;
    int j = size2 - 2;

    double Ref, Imf, Reg, Img;
    double real, imag;

    for (int i = 0; i < size2; i += 2) {
        if (j < i) break;

        // Unpack F and G from real/imag
        Ref = 0.5 * (data[i] + data[j]);
        Imf = 0.5 * (data[i+1] - data[j+1]);

        Reg = 0.5 * (data[i+1] + data[j+1]);
        Img = 0.5 * (data[i] - data[j]);

        // Multiply (F)(G)
        real = Ref * Reg - Imf * Img;
        imag = Ref * Img + Imf * Reg;

        data[i]   = real;
        data[i+1] = imag;

        if (i>0){
            data[j]   = real;
            data[j+1] = -imag;
            j -= 2;
        }
    }

}


void cpu_convolve(double *QT,
                const double* series_padded, 
                const double* Q_padded, 
                const int padded_size,
                const int interval_size,
                const int mp_size){

    // Writing complex vector
    size_t N = get_next_power_of_two(padded_size);
    std::vector<double> two_inpts(2*N,0.0f);

    for(int i=0; i<padded_size;i++){
        two_inpts[2*i] = series_padded[i];
        two_inpts[2*i+1] = Q_padded[i];
    }


    // FFT
    inplace_fft_Z2Z(two_inpts.data(),N,1);

    // Multiply FFTs
    inplace_mult_fft(two_inpts.data(),N);

    // IFFT
    inplace_fft_Z2Z(two_inpts.data(),N,-1);

    // Geeting real part of multiplication
    for(int i=0; i<mp_size;i++){
        QT[i] = two_inpts.at(2*(i+interval_size-1))/1;
    }

}