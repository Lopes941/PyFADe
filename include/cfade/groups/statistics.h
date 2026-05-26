#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>
#include <cfade/groups/groups.h>
#include <cfade/features/features.h>
#include <cfade/features/statistics.h>

#include <vector>
#include <memory>
#include <string>
#include <any>
#include <stdexcept>
#include <variant>

namespace cfade
{
    
    /**
     * @brief Class for computing continuous statistics (mean and standard deviation).
     * 
     * This class is a feature group that computes the mean and standard deviation
     * of continuous data in a dataset. It inherits from IFeatureGroupCRTP to provide
     * a common interface for feature groups.
     * 
     * It contains two main features:
     * - ContinuousMean: Computes the mean of the continuous data.
     * - ContinuousStandardDeviation: Computes the standard deviation of the continuous data.   
     * 
     * @tparam Derived The derived class that implements the feature group.
     */
    class ContinuousStatistics: public IFeatureGroupCRTP<ContinuousStatistics>{

            private:

                /**
                 * @brief Group name for the ContinuousStatistics feature group.
                 * 
                 * This static constant holds the name of the feature group, which is used to identify it in the system.
                 */
                inline static const std::string group_name = "ContinuousStatistics";

                /**
                 * @brief Feature names for the ContinuousStatistics feature group.
                 * 
                 * This static constant holds the names of the features in the group, which are used to identify them in the system.
                 */
                inline static const std::vector<std::string> feature_names_string = {
                    ContinuousMean::static_name(),
                    ContinuousStandardDeviation::static_name()};

                /**
                 * @brief Requirements for the ContinuousStatistics feature group.
                 * 
                 * This static constant holds the names of the required feature groups for this feature group.
                 * Currently, there are no specific requirements.
                 * 
                 */
                inline static const std::vector<std::string> requirements_string = {};

                /**
                 * @brief Parameters for the ContinuousStatistics feature group.
                 * 
                 * This static constant holds the names of the parameters for this feature group.
                 * Currently, there are no specific parameters.
                 */
                inline static const std::vector<std::string> parameters_string = {};


                /**
                 * @brief Mean feature for continuous data.
                 * 
                 * This shared pointer holds the ContinuousMean feature, which computes the mean of the continuous data.
                 */
                std::shared_ptr<ContinuousMean> means;

                /**
                 * @brief Standard deviation feature for continuous data.
                 * 
                 * This shared pointer holds the ContinuousStandardDeviation feature, which computes the standard deviation of the continuous data.
                 */
                std::shared_ptr<ContinuousStandardDeviation> stddev;

            public:

                /**
                 * @brief Get the name of the feature group.
                 * 
                 * This method returns the name of the feature group, which is used to identify it in the system.
                 * 
                 * @return const std::string& The name of the feature group.
                 */
                static const std::string& static_name() { return group_name;}

                /**
                 * @brief Get the names of the features in the group.
                 * 
                 * This method returns a vector of strings containing the names
                 * of the features in the group, which are used to identify them in the system.
                 * 
                 * @return const std::vector<std::string>& Names of the features in the group.
                 */
                static const std::vector<std::string>& static_feature_names() { return feature_names_string;}

                /**
                 * @brief Get the requirements for the feature group.
                 * 
                 * This method returns a vector of strings containing the names
                 * of the required feature groups for this feature group.
                 * 
                 * @return const std::vector<std::string>& Requirements for the feature group.
                 */
                static const std::vector<std::string>& static_requirements() { return requirements_string;}

                /**
                 * @brief Get the parameters for the feature group.
                 * 
                 * This method returns a vector of strings containing the names
                 * of the parameters for this feature group.
                 * 
                 * @return const std::vector<std::string>& Parameters for the feature group.
                 */
                static const std::vector<std::string>& static_parameters() { return parameters_string;}

                /**
                 * @brief Default constructor for ContinuousStatistics.
                 * 
                 * This constructor initializes the feature group with zero rows.
                 */
                ContinuousStatistics()=default;

                /**
                 * @brief Constructor for ContinuousStatistics with a dataset and feature groups.
                 * 
                 * This constructor initializes the feature group with the provided dataset and feature groups.
                 * It sets up the means and standard deviation features based on the dataset.
                 * 
                 * @param observed_dataset The dataset containing the observed data.
                 * @param requirements The feature groups that are required for this feature group.
                 */
                ContinuousStatistics(const std::shared_ptr<DataSet>& observed_dataset,
                const std::vector<std::shared_ptr<IFeatureGroup>>& requirements);

                /**
                 * @brief Default destructor for ContinuousStatistics.
                 * 
                 * This destructor is defaulted and does not perform any additional actions.
                 */
                ~ContinuousStatistics()=default;

                /**
                 * @brief Update the statistics with new data.
                 * 
                 * This method updates the means and standard deviation features based on the new data.
                 * 
                 * @param observed_dataset The dataset containing the observed data to update the statistics.
                 * @param window_size The size of the window to consider for the updates.
                 */
                void update(const std::shared_ptr<DataSet>& observed_dataset, int window_size) override;

        };

} // namespace cfade