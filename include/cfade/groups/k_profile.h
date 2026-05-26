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

    /**
     * @brief K-Dimensional Matrix profile feature group.
     * 
     * This class represents the k-dimensional matrix profile feature group.
     * The k-dimensional matrix profile is an aggregation technique for multi-dimensional
     * time-series, returning a data structure that incorporates the matrix profiles of
     * each dimension of the time series.
     *  
     * It is used for tasks such as motif discovery, anomaly detection, and similarity search.
     * This class inherits from IFeatureGroupCRTP and implements the necessary methods for a feature group.
     * 
     * @tparam Derived The derived class that implements the feature group.
     */
    class KProfile : public IFeatureGroupCRTP<KProfile>{

        private:
        
            /**
             * @brief Group name for the KProfile feature group.
             * 
             * This static constant holds the name of the feature group, which is used to identify it in the system.
             */
            inline static const std::string group_name = "KProfile";

            /**
             * @brief Feature names for the KProfile feature group.
             * 
             * This static constant holds the names of the features in the group, which are used to identify them in the system.
             */
            inline static const std::vector<std::string> feature_names_string = {
                    KProfileValue::static_name(),
                    KProfileIndex::static_name()};

            /**
             * @brief Requirements for the KProfile feature group.
             * 
             * This static constant holds the names of the required feature groups for this feature group.
             * Currently, it requires the MatrixProfile feature group.
             */
            inline static const std::vector<std::string> requirements_string = {MatrixProfile::static_name()};

            /**
             * @brief Parameters for the KProfile feature group.
             * 
             * This static constant holds the names of the parameters for this feature group.
             * Currently, there are no specific parameters.
             */
            inline static const std::vector<std::string> parameters_string = {};

            std::shared_ptr<KProfileValue> k_profile; /**< Values of the k-dimensional matrix profile */
            std::shared_ptr<KProfileIndex> k_index; /**< Indices of the k-dimensional matrix profile */

            std::shared_ptr<VectorGroup<double>> matrix_profile; /**< Matrix profile values */
            std::shared_ptr<VectorGroup<int>> mp_index; /**< Indices of the matrix profile */


        public:

            /**
             * @brief Get the name of the feature group.
             * 
             * This static method returns the name of the feature group, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature group.
             */
            static const std::string& static_name() { return group_name;}

            /**
             * @brief Get the names of the features in the group.
             * 
             * This static method returns a vector of strings containing the names
             * of the features in the group. It is expected that each derived class
             * will implement this method to return the appropriate feature names.
             * 
             * @return const std::vector<std::string>& Names of the features in the group.
             */
            static const std::vector<std::string>& static_feature_names() { return feature_names_string;}

            /**
             * @brief Get the requirements of the feature group.
             * 
             * This static method returns a vector of strings containing the names
             * of the required feature groups for this feature group. It is expected
             * that each derived class will implement this method to return the
             * appropriate requirements.
             * 
             * @return const std::vector<std::string>& Requirements of the feature group.
             */
            static const std::vector<std::string>& static_requirements() { return requirements_string;}

            /**
             * @brief Get the parameters of the feature group.
             * 
             * This static method returns a vector of strings containing the names
             * of the parameters for this feature group. It is expected that each
             * derived class will implement this method to return the appropriate
             * parameters.
             * 
             * @return const std::vector<std::string>& Parameters of the feature group.
             */
            static const std::vector<std::string>& static_parameters() { return parameters_string;}

            /**
             * @brief Default constructor for KProfile.
             * 
             * This constructor initializes the feature group with zero rows.
             */
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

            /**
             * @brief Default destructor for MatrixProfile.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~KProfile()=default;

            /**
             * @brief Update the feature group with new data.
             * 
             * This method updates the feature group with new data from the observed dataset.
             * It is expected that each derived class will implement this method to perform
             * the appropriate updates.
             * 
             * @param observed_dataset The dataset to use for the update.
             * @param window_size The size of the window to use for the update.
             */
            void update(const std::shared_ptr<DataSet>& observed_dataset, int window_size) override;
    };

} // namespace cfade