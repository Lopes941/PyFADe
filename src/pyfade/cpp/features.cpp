
#include <cfade/features/features.h>
#include <cfade/features/statistics.h>
#include <cfade/features/k_profile.h>
#include <cfade/features/matrix_profile.h>

#include <unordered_map>
#include <typeindex>
#include <typeinfo>
#include <memory>

namespace cfade{

    // std::unordered_map<std::string, std::type_index> feature_type_map = {
    //     {
    //         ContinuousMean::static_name(), std::type_index(typeid(ContinuousMean::value_type)),
    //     },
    //     {
    //         ContinuousStandardDeviation::static_name(), std::type_index(typeid(ContinuousStandardDeviation::value_type)),
    //     },
    //     {
    //         MatrixProfileValue::static_name(), std::type_index(typeid(MatrixProfileValue::value_type)),
    //     },
    //     {
    //         MatrixProfileIndex::static_name(), std::type_index(typeid(MatrixProfileIndex::value_type)),
    //     },
    //     {
    //         KProfileValue::static_name(), std::type_index(typeid(KProfileValue::value_type)),
    //     },
    //     {
    //         KProfileIndex::static_name(), std::type_index(typeid(KProfileIndex::value_type)),
    //     }

    // };

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