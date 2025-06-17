#pragma once

#include <vector>
#include <memory>
#include <string>
#include <any>
#include <stdexcept>
#include <variant>


#include <cpp/utils.h>
#include <cpp/dataset.h>
#include <cpp/window.h>
#include <cpp/features.h>

namespace cfade
{

    // ===============
    // Features
    // ===============

    

    class ContinuousStandardDeviation: public IFeatureCRTP<ContinuousStandardDeviation>{

        public:

            using value_type = double;
            using IFeatureCRTP<ContinuousStandardDeviation>::IFeatureCRTP;

            inline static const std::string feature_name = "stddev";
            static const std::string& static_name() { return feature_name;}
            
            ~ContinuousStandardDeviation()=default;

            inline double at(int row, int col) const{
                return IFeatureCRTP<ContinuousStandardDeviation>::at<typename value_type>(row,col);
            }

            inline double& at(int row, int col){
                return IFeatureCRTP<ContinuousStandardDeviation>::at<typename value_type>(row,col);
            }

    };

    class ContinuousStatistics: public IFeatureGroupCRTP<ContinuousStatistics>{

            private:

                inline static const std::string group_name = "ContinuousStatistics";
                inline static const std::vector<std::string> feature_names_string = {
                    ContinuousMean::static_name(),
                    ContinuousStandardDeviation::static_name()};
                inline static const std::vector<std::string> requirements_string = {};
                inline static const std::vector<std::string> parameters_string = {};


                std::shared_ptr<ContinuousMean> means;
                std::shared_ptr<ContinuousStandardDeviation> stddev;

            public:

                static const std::string& static_name() { return group_name;}
                static const std::vector<std::string>& static_feature_names() { return feature_names_string;}
                static const std::vector<std::string>& static_requirements() { return requirements_string;}
                static const std::vector<std::string>& static_parameters() { return parameters_string;}

                ContinuousStatistics()=default;
                ContinuousStatistics(const std::shared_ptr<DataSet>&,
                    const std::vector<std::shared_ptr<IFeatureGroup>>);
                ~ContinuousStatistics()=default;

                void update(const std::shared_ptr<DataSet>, int) override;
                
        };

} // namespace cfade