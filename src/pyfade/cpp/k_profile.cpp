
#include <cfade/utils/utils.h>
#include <cfade/groups/groups.h>
#include <cfade/groups/k_profile.h>
#include <cfade/features/features.h>
#include <cfade/features/k_profile.h>

#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>

namespace cfade{

    KProfile::KProfile(const std::shared_ptr<DataSet>& observed_dataset,
                        const std::vector<std::shared_ptr<IFeatureGroup>>& dependencies):
    k_profile(std::make_shared<KProfileValue>(observed_dataset->get_dimension())),
    k_index(std::make_shared<KProfileIndex>(observed_dataset->get_dimension()))
    {
        bool found = false;
        for(auto dependency : dependencies){
            if(dependency->name() == MatrixProfile::static_name()){
                matrix_profile = std::get<std::shared_ptr<VectorGroup<double>>>(dependency->get_feature(MatrixProfileValue::static_name()));
                mp_index = std::get<std::shared_ptr<VectorGroup<int>>>(dependency->get_feature(MatrixProfileIndex::static_name()));
                found = true;
                break;
            }
        }

        if (!found){
            throw std::runtime_error("Did not find " + MatrixProfile::static_name() + " dependency from " + static_name());
        }

        feature_map[KProfileValue::static_name()] = k_profile;
        feature_map[KProfileIndex::static_name()] = k_index;
        
    }

    void KProfile::update(const std::shared_ptr<DataSet>& observed_dataset,
                        int window_size){
        
        int start_loc = k_profile->get_length();
        int final_size = matrix_profile->cols;
        int total_dim = observed_dataset->get_dimension();

        k_profile->increase_cols(final_size);
        k_index->increase_cols(final_size);

        std::vector<int> ind_sort;
        for(int col=start_loc; col<final_size; col++){

            ind_sort = argsort_descending(
                matrix_profile->data.data() + matrix_profile->rows*col, 
                matrix_profile->rows);

            for(int dim=0; dim<total_dim; dim++){
                k_index->at(dim,col) = ind_sort[dim];
                k_profile->at(dim,col) = matrix_profile->at(ind_sort[dim],col);
            }
        }
    }

} // namespace cfade