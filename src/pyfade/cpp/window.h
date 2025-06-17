#pragma once

#include <vector>
#include <memory>
#include <string>
#include <any>
#include <stdexcept>
#include <variant>
#include <map>

#include<cpp/dataset.h>
#include <cpp/features.h>


namespace cfade{
    

    



    // =====================
    // Parameter definitions
    // =====================


    class IFeatureGroupParameter{

        public:
            ~IFeatureGroupParameter()=default;
            virtual const std::string& name() const = 0;
            virtual void set(VariantTypes)=0;
            
    };

    template <typename Derived>
    class FeatureGroupParameter: public IFeatureGroupParameter{

        private:
            VariantTypes value;

        public:

            ~FeatureGroupParameter()=default;

            inline const std::string& name() const override final{
                return Derived::static_name();
            }

            inline void set(VariantTypes val) override final{
                value = std::visit(
                    [](auto&& arg) -> typename Derived::value_type {
                        return static_cast<typename Derived::value_type>(arg);
                    },val);
            }

            inline auto get() const{
                return std::get<typename Derived::value_type>(value);
            }
    };



    


} // namespace cfade









    // class DiscreteStatistics: public IFeatureGroup{

        

    //     private:
        
    //         VectorGroup<double> median;
    //         VectorGroup<double> quantile;

    //         const VectorGroup<double>& get_medians() const;
    //         const VectorGroup<double>& get_quantiles() const;
        
    //     public:

    //         inline static const std::string group_name = "DiscreteStatistics";
    //         inline static const std::vector<std::string> feature_names_string = {"median","quantile"};

    //         inline const std::string& name() const override final { return group_name;}
    //         inline const std::vector<std::string>& feature_names() const override final { return feature_names_string;}

    //         DiscreteStatistics()=default;
    //         DiscreteStatistics(const std::shared_ptr<DataSet>&);
    //         ~DiscreteStatistics()=default;

    //         void update(const std::shared_ptr<DataSet>, int) override;

    //         const VectorGroup<double>& get_feature(const std::string&) const override;
            

    // };
