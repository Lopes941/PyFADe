#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace cfade
{

    class Mat_Profile{

    private:

        bool left_only=true;
        bool started_runtime=false;

        int skip_start=0;


        int append_size          = 0;
        int series_size          = 0;
        int interval_size        = 0;
        int mp_size              = 0;
        int current_QT_location  = 0;
        int exclusion_zone_size  = 0;

        float exclusion_zone_ratio  = 0.5;


        std::vector<int> mp_index   = std::vector<int>(0);

        std::vector<double> series      = std::vector<double>(0);
        std::vector<double> QT          = std::vector<double>(0);
        std::vector<double> mat_profile = std::vector<double>(0);
        std::vector<double> means       = std::vector<double>(0);
        std::vector<double> stds        = std::vector<double>(0);
        

        // CUDA pointers
        // int* d_ind=nullptr;
        double* d_series=nullptr;
        double* d_means=nullptr;
        double* d_stds=nullptr;
        double* d_QT=nullptr;
        // double* d_mp=nullptr;
        
        

    public:

        Mat_Profile();
        Mat_Profile(const std::vector<double>);
        Mat_Profile(const std::vector<double>, const int);
        ~Mat_Profile();

        // Parameter setters
        void set_interval_size(const int);
        void set_exclusion_ratio(const float);
        void set_start_ignore(const int);
        void set_left_only(const bool);

        // Parameter getters
        double get_size() const;
        std::vector<int> get_mp_ind() const;
        std::vector<double> get_series() const;
        std::vector<double> get_mp() const;
        std::vector<double> get_QT() const;
        std::vector<double> get_means() const;
        std::vector<double> get_stds() const;

        // CUDA operations
        void initialize_cuda();
        void release_from_cuda();
        void get_first_product(int start);


        
        // Run operations
        void run_batch();
        void init_runtime();
        void stop_runtime();

        // Including data
        void append_data(std::vector<double>);
        void reserve_mp(size_t incremented_size);
        

    };

    std::vector<double> calculate_means(const std::vector<double>& series, int interval_size);
    std::vector<double> calculate_stds(const std::vector<double>& series, int interval_size, const std::vector<double>& means);
    
} // namespace cfade
