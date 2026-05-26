
#include <cfade/features/features.h>
#include <cfade/features/statistics.h>
#include <cfade/features/k_profile.h>
#include <cfade/features/matrix_profile.h>

#include <unordered_map>
#include <typeindex>
#include <typeinfo>
#include <memory>

namespace cfade{


    std::unordered_map<std::string, FeatureFactory> feature_builder_map = {
        {
            ContinuousMean::static_name(),
            [](int row){
                return std::make_shared<ContinuousMean>(row);
            },
        },
        {
            ContinuousStandardDeviation::static_name(),
            [](int row){
                return std::make_shared<ContinuousStandardDeviation>(row);
            },
        },
        {
            MatrixProfileValue::static_name(),
            [](int row){
                return std::make_shared<MatrixProfileValue>(row);
            },
        },
        {
            MatrixProfileIndex::static_name(), 
            [](int row){
                return std::make_shared<MatrixProfileIndex>(row);
            },
        },
        {
            KProfileValue::static_name(), 
            [](int row){
                return std::make_shared<KProfileValue>(row);
            },
        },
        {
            KProfileIndex::static_name(),
            [](int row){
                return std::make_shared<KProfileIndex>(row);
            },
        }

    };

} // namespace cfade