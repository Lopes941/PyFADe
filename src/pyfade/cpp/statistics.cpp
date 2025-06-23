
#include <cfade/features/statistics.h>
#include <cfade/features/features.h>
#include <cfade/dataset/dataset.h>
#include <cfade/utils/utils.h>
#include <cfade/groups/statistics.h>

#include <numeric>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <string>

namespace cfade
{

    ContinuousStatistics::ContinuousStatistics(const std::shared_ptr<DataSet>& observed_dataset,
    const std::vector<std::shared_ptr<IFeatureGroup>>& requirements):
    IFeatureGroupCRTP(observed_dataset->get_dimension()){

        if (observed_dataset->get_length() < 1){
            throw std::invalid_argument("Observed dataset must have at least one data point.");
        }
        if (observed_dataset->get_dimension() < 1){
            throw std::invalid_argument("Observed dataset must have at least one dimension.");
        }

        means = std::dynamic_pointer_cast<ContinuousMean>(feature_map[ContinuousMean::static_name()]);
        stddev = std::dynamic_pointer_cast<ContinuousStandardDeviation>(feature_map[ContinuousStandardDeviation::static_name()]);
        
    }

    void ContinuousStatistics::update(const std::shared_ptr<DataSet>& observed_dataset, int window_size){

        if (observed_dataset->get_length() < window_size){
            return;
        }

        int last_means_size = means->get_length();
        int added_size = observed_dataset->get_length() - (last_means_size+window_size-1);

        std::vector<double> new_cumsum(added_size+window_size);
        std::vector<double> new_cumsum2(added_size+window_size);

        means->increase_cols(added_size);
        stddev->increase_cols(added_size);

        for(int dimension=0; dimension<observed_dataset->get_dimension(); dimension++){

            for(int i=0; i<added_size+window_size-1; i++){
                double current_data = observed_dataset->get_data(dimension,last_means_size+i);
                new_cumsum[i+1] = current_data + new_cumsum[i];
                new_cumsum2[i+1] = current_data*current_data + new_cumsum2[i];

                if(i>=window_size-1){
                    double mean_val = (new_cumsum[i+1] - new_cumsum[i-window_size+1])/window_size;

                    means->at(dimension,last_means_size+i-window_size+1) = mean_val;
                    stddev->at(dimension,last_means_size+i-window_size+1) = 
                        std::sqrt((new_cumsum2[i+1] - new_cumsum2[i-window_size+1])/window_size - 
                        mean_val*mean_val);
                }
            }
        }
    }

    
    // DiscreteStatistics::DiscreteStatistics(const std::shared_ptr<DataSet>& observed_dataset):
    //     median(VectorGroup<double>(observed_dataset->get_dimension())),
    //     quantile(VectorGroup<double>(observed_dataset->get_dimension()))
    // {}

    // void DiscreteStatistics::update(const std::shared_ptr<DataSet> observed_dataset, int window_size){

    //     if (observed_dataset->get_length() < window_size){
    //         return;
    //     }

    //     int last_median_size = median.cols;
    //     int added_size = observed_dataset->get_length() - (last_median_size+window_size-1);
    //     int median_index = window_size/2;

    //     std::function<double(const std::vector<int>&,int&)> median_getter;
    //     if (window_size%2==0){
    //         median_getter = [&median_index,&last_median_size,&observed_dataset](const std::vector<int>& sorted_index, int& dimension){
    //             return (observed_dataset->get_data(dimension,last_median_size+sorted_index[median_index-1]) +
    //                 observed_dataset->get_data(dimension,last_median_size+sorted_index[median_index]))/2;
    //             };
    //     }else{
    //         median_getter = [&median_index,&last_median_size,&observed_dataset]
    //         (const std::vector<int>& sorted_index, int& dimension){
    //             return observed_dataset->get_data(dimension,last_median_size+sorted_index[median_index-1]);
    //             };
    //     }

        

    //     std::vector<double> sorted_vector(window_size);
    //     std::vector<int> sorted_index(window_size);
    //     for(int dimension=0; dimension<observed_dataset->get_dimension(); dimension++){
    //         median.increase_cols(added_size);
    //         quantile.increase_cols(added_size);

    //         for(int k=0;k<window_size;k++){
    //             sorted_index[k] = k;
    //             sorted_vector[k] = observed_dataset->get_data(dimension,last_median_size+k);
    //         }

    //         std::sort(sorted_index.begin(),
    //             sorted_index.end(),
    //             [&sorted_vector](int i1, int i2){return sorted_vector[i1]<sorted_vector[i2];});

    //         median.at(dimension,last_median_size) = median_getter(sorted_index,dimension);

    //         for(int k=1;k<added_size;k++){

    //             int index = last_median_size+k;

    //             double new_val = observed_dataset->get_data(dimension,last_median_size+window_size+k-1);
    //             for(int update_ind=0;update_ind<window_size;update_ind++){
    //                 if(replace_forward){
    //                     sorted_index[update_ind] = sorted_index[update_ind+1];
    //                 }

    //             }

    //             median.at(dimension,index);
    //         }


    //     }
    // }


    // const VectorGroup<double>& DiscreteStatistics::get_feature(const std::string& feature)const{

    //     if(feature==feature_names_string[0]){
    //         return get_medians();
    //     }else if(feature == feature_names_string[1]){
    //         return get_quantiles();
    //     }else{
    //         throw std::runtime_error("Invalid feature");
    //     }
    // }

    // const VectorGroup<double>& DiscreteStatistics::get_medians()const{
    //     return median;
    // }

    // const VectorGroup<double>& DiscreteStatistics::get_quantiles()const{
    //     return quantile;
    // }


} // namespace cfade
