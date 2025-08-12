#include <iostream>
#include <vector>
#include <thread>
#include <cmath>

using ComplexArray = std::vector<double>;

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
    constexpr double PI2 = 6.28318530717958647692;

    while (mmax < size2) {
        int istep = mmax << 1;
        double theta = isign * ( PI2 / mmax);  // 2π/mmax

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


void realft(std::vector<double>& data, 
            size_t size,
            int isign=1){

    int i, i1, i2, i3, i4;
    int n = data.size();

    double c1=0.5;
    double c2, h1r, h1i, h2r, h2i, wr, wi, wpr, wpi, wtemp;
    double theta = 3.141592653589793238 / static_cast<double>(n>>1);

    if (isign == 1) {
        c2 = -0.5;
        inplace_fft_Z2Z(data, size, isign);
    } else {
        c2 = 0.5;
        theta = -theta;
    }

    wtemp = std::sin(0.5 * theta);
    wpr = -2.0 * wtemp * wtemp;
    wpi = std::sin(theta);
    wr = 1.0 + wpr;
    wi = wpi;

    for (int k = 1; k < size / 2; ++k) {
        int i1 = 2 * k;
        int i2 = 2 * (size - k);

        h1r = c1 * (data[i1] + data[i2]);
        h1i = c1 * (data[i1 + 1] - data[i2 + 1]);
        h2r = -c2 * (data[i1 + 1] + data[i2 + 1]);
        h2i = c2 * (data[i1] - data[i2]);

        data[i1]     = h1r + wr * h2r - wi * h2i;
        data[i1 + 1] = h1i + wr * h2i + wi * h2r;
        data[i2]     = h1r - wr * h2r + wi * h2i;
        data[i2 + 1] = -h1i + wr * h2i + wi * h2r;

        double wtemp_copy = wr;
        wr = wtemp_copy * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp_copy * wpi + wi;
    }

    if (isign == 1) {
        double tmp = data[0];
        data[0] = tmp + data[1];
        data[1] = tmp - data[1];
    } else {
        double tmp = data[0];
        data[0] = 0.5 * (tmp + data[1]);
        data[1] = 0.5 * (tmp - data[1]);
    }

}


// Inplace fft (based on numerical recipes)
void inplace_mult_fft(ComplexArray& data, size_t size) {
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

void print_arr(ComplexArray& arr, size_t N){
    for (int k = 0; k <= N/2; ++k) {
        double re, im;
        if (k == 0) {
            re = arr[0];
            im = 0.0;
        } else if (k == N/2) {
            re = arr[1];
            im = 0.0;
        } else {
            re = arr[2*k];
            im = arr[2*k+1];
        }
        std::cerr << re << " + " << im << " i, ";
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
    std::cerr << N << " , " << padded_size << std::endl;

    ComplexArray p_series(2*N,0.0);
    ComplexArray p_query(2*N,0.0);
    ComplexArray two_inpts(2*N,0.0f);

    for(int i=0; i<padded_size;i++){
        p_series[2*i] = series_padded[i];
        p_query[2*i] = Q_padded[i];
    }
    
    


    // FFT
    // inplace_fft_Z2Z(p_series,N,1);
    realft(p_series,N,1);
    std::cerr << "Series:" << std::endl;
    // print_arr(p_series,padded_size);
    for(int i=1; i<2*N; i+=2){
        std::cerr << p_series[i-1] << " + " << p_series[i] << " i, ";
    }
    std::cerr << std::endl;

    // inplace_fft_Z2Z(p_query,N,1);
    realft(p_query,N,1);
    std::cerr << "Query:" << std::endl;
    for(int i=1; i<2*N; i+=2){
        std::cerr << p_query[i-1] << " + " << p_query[i] << " i, ";
    }
    std::cerr << std::endl;

    // Multiply FFTs
    inplace_mult_fft(two_inpts,N);

    std::cerr << "Mult:" << std::endl;
    for(int i=1; i<2*N; i+=2){
        std::cerr << two_inpts[i-1] << " + " << two_inpts[i] << " i, ";
    }
    std::cerr << std::endl;

    // IFFT
    // inplace_fft_Z2Z(two_inpts,N,-1);

    // Geeting real part of multiplication
    for(int i=0; i<mp_size;i++){
        QT[i] = two_inpts[2*(i+interval_size-1)]/N;
    }

}