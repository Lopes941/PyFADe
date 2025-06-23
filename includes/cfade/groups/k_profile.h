#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>
#include <cfade/groups/groups.h>
#include <cfade/groups/matrix_profile.h>
#include <cfade/features/features.h>
#include <cfade/features/k_profile.h>

#include <vector>
#include <memory>
#include <string>

namespace cfade{

    class KProfile : public IFeatureGroupCRTP<KProfile>{

        private:
        
            inline static const std::string group_name = "KProfile";
            inline static const std::vector<std::string> feature_names_string = {
                    KProfileValue::static_name(),
                    KProfileIndex::static_name()};
            inline static const std::vector<std::string> requirements_string = {MatrixProfile::static_name()};
            inline static const std::vector<std::string> parameters_string = {
                UseCudaParam::parameter_name,
                LeftOnlyParam::parameter_name,
                SkipStartParam::parameter_name,
                ExclusionZoneRatioParam::parameter_name,
            };

            std::shared_ptr<KProfileValue> k_profile;
            std::shared_ptr<KProfileIndex> k_index;

            std::shared_ptr<VectorGroup<double>> matrix_profile;
            std::shared_ptr<VectorGroup<int>> mp_index;


        public:

            static const std::string& static_name() { return group_name;}
            static const std::vector<std::string>& static_feature_names() { return feature_names_string;}
            static const std::vector<std::string>& static_requirements() { return requirements_string;}
            static const std::vector<std::string>& static_parameters() { return parameters_string;}

            KProfile()=default;

            /**
             * @brief Constructor for KProfile.
             * 
             * Initializes the KProfile with the observed dataset and requirements.
             * 
             * @param observed_dataset The dataset to be observed.
             * @param requirements The feature groups required for this profile.
             */
            KProfile(const std::shared_ptr<DataSet>& observed_dataset, 
                const std::vector<std::shared_ptr<IFeatureGroup>>& requirements);
            ~KProfile()=default;

            void update(const std::shared_ptr<DataSet>& observed_dataset, int window_size) override;
    };

} // namespace cfade