#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace cfade
{


    enum class Property{
        LEFT_ONLY
    };


    class Mat_Profile{

    private:

        bool left_only=false;

        size_t skip_start=0;


        size_t series_size          = 0;
        size_t interval_size        = 0;
        size_t mp_size              = 0;
        size_t current_QT_location  = 0;
        size_t exclusion_zone_size  = 15;

        float exclusion_zone_ratio  = 0.5;


        std::vector<size_t> mp_index   = std::vector<size_t>(0);

        std::vector<double> series      = std::vector<double>(0);
        // std::vector<double> QT          = std::vector<double>(0);
        // std::vector<double> QT_old      = std::vector<double>(0);
        std::vector<double> mat_profile = std::vector<double>(0);
        std::vector<double> means       = std::vector<double>(0);
        std::vector<double> stds        = std::vector<double>(0);
        

        // CUDA pointers
        size_t* d_ind=nullptr;
        double* d_series=nullptr;
        double* d_means=nullptr;
        double* d_stds=nullptr;
        double* d_QT_even=nullptr;
        double* d_QT_odd=nullptr;
        double* d_QT_first=nullptr;
        double* d_mp=nullptr;
        
        // double* d_series=nullptr;
        

    public:

        Mat_Profile(const std::vector<double>);
        Mat_Profile(const std::vector<double>, const size_t);
        ~Mat_Profile();

        // Parameter setters
         void set_interval_size(const size_t);

        // Parameter getters
        double return_size() const;
        std::vector<size_t> get_mp_ind() const;
        std::vector<double> get_series() const;
        std::vector<double> get_mp() const;
        std::vector<double> get_means() const;
        std::vector<double> get_stds() const;

        // CUDA operations
        void initialize_cuda();
        void get_data_from_cuda();
        void get_first_product();


        
        

        void run_batch();
    };

    std::vector<double> calculate_means(const std::vector<double>& series, size_t interval_size);
    std::vector<double> calculate_stds(const std::vector<double>& series, size_t interval_size, const std::vector<double>& means);
    
    
    void size_checker(size_t size);
    
    
} // namespace cfade
