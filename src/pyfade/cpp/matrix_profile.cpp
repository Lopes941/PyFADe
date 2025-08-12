#include <cfade/utils/utils.h>
#include <cfade/groups/groups.h>
#include <cfade/groups/matrix_profile.h>
#include <cfade/features/features.h>
#include <cfade/features/matrix_profile.h>
#include <cfade/core/cpu_funcs.h>

#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <variant>
#include <algorithm>
#include <numeric>

#include <iostream>


#ifdef USE_CUDA
    #include <cfade/cuda/cuda_funcs.h>
    #include <cufft.h>
    #include <cuda_runtime.h>
#endif


namespace cfade{

    std::shared_ptr<QTDataInterface> get_QT(const std::shared_ptr<DataSet> observed_dataset,
                            const std::shared_ptr<VectorGroup<double>> means,
                            const std::shared_ptr<VectorGroup<double>> stds,
                            int window_size,
                            bool use_cuda);

    void run_iterations(const std::shared_ptr<cfade::DataSet> observed_dataset,
                        const std::shared_ptr<cfade::VectorGroup<double>> means,
                        const std::shared_ptr<cfade::VectorGroup<double>> stds,
                        std::shared_ptr<QTDataInterface> QT,
                        std::shared_ptr<cfade::VectorGroup<double>> matrix_profile, 
                        std::shared_ptr<cfade::VectorGroup<int>> index,
                        int start_loc, 
                        const int final_size, 
                        const int window_size,
                        const int exclusion_zone_size,
                        const bool left_only,
                        const bool use_cuda);



    MatrixProfile::MatrixProfile(const std::shared_ptr<DataSet>& observed_dataset,
                                const std::vector<std::shared_ptr<IFeatureGroup>>& dependencies):
    matrix_profile(std::make_shared<MatrixProfileValue>(observed_dataset->get_dimension())),
    index(std::make_shared<MatrixProfileIndex>(observed_dataset->get_dimension())),
    use_cuda(UseCudaParam(false)),
    left_only(LeftOnlyParam(false)),
    skip_start(SkipStartParam(-1)),
    exclusion_zone_ratio(ExclusionZoneRatioParam(0.5))
    {
        bool found = false;
        for(auto dependency : dependencies){
            if(dependency->name() == ContinuousStatistics::static_name()){
                means = std::get<std::shared_ptr<VectorGroup<double>>>(dependency->get_feature(ContinuousMean::static_name()));
                stds = std::get<std::shared_ptr<VectorGroup<double>>>(dependency->get_feature(ContinuousStandardDeviation::static_name()));
                found = true;
                break;
            }
        }

        if (!found){
            throw std::runtime_error("Did not find " + ContinuousStatistics::static_name() + " dependency from " + static_name());
        }
        raw_matrix_profile = std::make_shared<VectorGroup<double>>(observed_dataset->get_dimension());
        feature_map[MatrixProfileValue::static_name()] = matrix_profile;
        feature_map[MatrixProfileIndex::static_name()] = index;
        
    }

    void MatrixProfile::set_parameter(const std::string& name, VariantTypes val) {

        if (name == UseCudaParam::parameter_name){
            use_cuda.set(val);
        }else if (name == LeftOnlyParam::parameter_name){
            left_only.set(val);
        }else if (name == SkipStartParam::parameter_name){
            skip_start.set(val);
        }else if (name == ExclusionZoneRatioParam::parameter_name){
            exclusion_zone_ratio.set(val);
        }else if (name == QuantileParam::parameter_name){
            quantile_param.set(val);
        }else{
            std::string text = static_name() + " has no '" + name + "' parameter.";
            throw std::invalid_argument(text);
        }
    }

    void MatrixProfile::update(const std::shared_ptr<DataSet>& observed_dataset,
                        int window_size){

        if (observed_dataset->get_length() < window_size){
            return;
        }
        
        int start_loc = matrix_profile->get_length();

        int final_size = observed_dataset->get_length() - window_size + 1;

        if(skip_start.get()==-1)
            skip_start = SkipStartParam(window_size);

        if(final_size < skip_start.get()){
            return;
        }

        matrix_profile->increase_cols(final_size-start_loc);
        index->increase_cols(final_size-start_loc);
        raw_matrix_profile->increase_cols(final_size-start_loc);

        int exclusion_zone_size = (int) std::round(exclusion_zone_ratio.get()*window_size);

        for(int old_ind=0; old_ind<start_loc; old_ind++){
            for(int dim=0; dim<observed_dataset->get_dimension();dim++){
                matrix_profile->at(dim,old_ind) = raw_matrix_profile->at(dim,old_ind);
            }
        }
        
        std::shared_ptr<QTDataInterface> QT = get_QT(observed_dataset,
                                    means,
                                    stds,
                                    window_size,
                                    use_cuda.get());

        run_iterations(observed_dataset, 
                        means,
                        stds, 
                        std::move(QT), 
                        std::get<std::shared_ptr<VectorGroup<double>>>(matrix_profile->get()), 
                        std::get<std::shared_ptr<VectorGroup<int>>>(index->get()), 
                        start_loc, 
                        final_size, 
                        window_size,
                        exclusion_zone_size,
                        left_only.get(),
                        use_cuda.get());

        

        
        for(int i=start_loc; i<skip_start.get(); i++){
            for(int dim=0; dim<observed_dataset->get_dimension();dim++){
                matrix_profile->at(dim,i) = 0;
                index->at(dim,i) = 0;
            }
        }

        for(int i=start_loc; i<final_size; i++){
            for(int dim=0; dim<observed_dataset->get_dimension();dim++){
                raw_matrix_profile->at(dim,i) = matrix_profile->at(dim,i);
            }
        }

        std::vector<double> quantiles = quantile(raw_matrix_profile, quantile_param.get());
        for(int i=0; i<final_size; i++){
            for(int dim=0; dim<observed_dataset->get_dimension();dim++){
                if(matrix_profile->at(dim,i) >= quantiles[dim]){
                    matrix_profile->at(dim,i) -= quantiles[dim];
                }else{
                    matrix_profile->at(dim,i) = 0;
                }
            }
        }
        
    }


    void run_iterations(const std::shared_ptr<cfade::DataSet> observed_dataset,
                        const std::shared_ptr<cfade::VectorGroup<double>> means,
                        const std::shared_ptr<cfade::VectorGroup<double>> stds,
                        std::shared_ptr<QTDataInterface> QT,
                        std::shared_ptr<cfade::VectorGroup<double>> matrix_profile, 
                        std::shared_ptr<cfade::VectorGroup<int>> index,
                        int start_loc, 
                        const int final_size, 
                        const int window_size,
                        const int exclusion_zone_size,
                        const bool left_only,
                        const bool use_cuda){

        #ifdef USE_CUDA
        if(use_cuda){
            cuda_STOMP_iterations(observed_dataset, 
                            means,
                            stds, 
                            QT->d_QT, 
                            matrix_profile, 
                            index, 
                            start_loc, 
                            final_size, 
                            window_size,
                            exclusion_zone_size,
                            left_only);
            return;
        }
        #endif
        
        // TODO

    }

    std::shared_ptr<QTDataInterface> get_QT(const std::shared_ptr<DataSet> observed_dataset,
                            const std::shared_ptr<VectorGroup<double>> means,
                            const std::shared_ptr<VectorGroup<double>> stds,
                            int window_size,
                            bool use_cuda){

        #ifdef USE_CUDA
        if (use_cuda){
            return std::make_shared<QTDataCUDA>(observed_dataset, means, stds, window_size);
        }
        #endif
        // return std::make_shared<QTDataCPU>(observed_dataset, means, stds, window_size);
        
        // TODO

    }

    QTDataCPU::QTDataCPU(const std::shared_ptr<DataSet> observed_dataset,
                        const std::shared_ptr<VectorGroup<double>> means,
                        const std::shared_ptr<VectorGroup<double>> stds,
                        int window_size){

        double * ad_QT = new double[means->total_size()];
        cudaMalloc((void**) &d_QT, sizeof(double) * means->total_size());

        for (int i=0; i<means->total_size();i++){
            ad_QT[i] = 0;
        }

        // Getting flipped first interval padded
        int padded_size = observed_dataset->get_length() + window_size-1;
        int mp_size = means->cols;
        std::vector<double> Q_padded(padded_size,0.0f);
        std::vector<double> series_padded(padded_size,0.0f);

        for(int dimension=0; dimension<observed_dataset->get_dimension(); dimension++){

            for (int i=0; i<window_size; i++){
                Q_padded[i] = observed_dataset->at(dimension,window_size-1-i);
            }

            for(int i=0;i<observed_dataset->get_length(); i++){
                series_padded[i] = observed_dataset->at(dimension,i);
            }

            cpu_convolve(ad_QT+mp_size*dimension, series_padded.data(), Q_padded.data(), padded_size, window_size,mp_size);
        }

        cudaMemcpy(d_QT,ad_QT, sizeof(double)* means->total_size(), cudaMemcpyHostToDevice);
        delete[] ad_QT;
    }
                        

    QTDataCPU::~QTDataCPU(){
        // if(d_QT) delete[] d_QT;
        if(d_QT) cudaFree(d_QT);
    }

    #ifdef USE_CUDA
        QTDataCUDA::QTDataCUDA(const std::shared_ptr<DataSet> observed_dataset,
                                const std::shared_ptr<VectorGroup<double>> means,
                                const std::shared_ptr<VectorGroup<double>> stds,
                                int window_size){


            cudaMalloc((void**) &d_QT, sizeof(double) * means->total_size());

            // Getting flipped first interval padded
            int padded_size = observed_dataset->get_length() + window_size-1;
            int mp_size = means->cols;
            std::vector<double> Q_padded(padded_size,0.0f);
            std::vector<double> series_padded(padded_size,0.0f);

            for(int dimension=0; dimension<observed_dataset->get_dimension(); dimension++){

                for (int i=0; i<window_size; i++){
                    Q_padded[i] = observed_dataset->at(dimension,window_size-1-i);
                }

                for(int i=0;i<observed_dataset->get_length(); i++){
                    series_padded[i] = observed_dataset->at(dimension,i);
                }

                cuda_convolve(d_QT+mp_size*dimension, series_padded.data(), Q_padded.data(), padded_size, window_size,mp_size);

            }
        }

        QTDataCUDA::~QTDataCUDA(){

            if(d_QT) cudaFree(d_QT);

        }

        

    #endif

} // namespace cfade