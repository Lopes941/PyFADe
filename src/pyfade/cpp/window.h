#pragma once

#include <vector>
#include <memory>
#include <string>

#include<cpp/dataset.h>


namespace cfade
{

    class FeatureGroupInterface{

        public:
            ~FeatureGroupInterface()=default;
            virtual const std::shared_ptr<VectorGroup<double>> get_feature(const std::string&) const=0;
            virtual const std::string& name() const = 0;
            virtual const std::vector<std::string>& feature_names() const = 0;
            virtual void update(const std::shared_ptr<DataSet>, int)=0;

    };



    class ContinuousStatistics: public FeatureGroupInterface{

        private:
            std::shared_ptr<VectorGroup<double>> means;
            std::shared_ptr<VectorGroup<double>> stds;
        
        public:

            inline static const std::string group_name = "ContinuousStatistics";
            inline static const std::vector<std::string> feature_names_string = {"mean","std"};

            inline const std::string& name() const override final { return group_name;}
            inline const std::vector<std::string>& feature_names() const override final { return feature_names_string;}

            ContinuousStatistics()=default;
            ContinuousStatistics(const std::shared_ptr<DataSet>&);
            ~ContinuousStatistics()=default;

            void update(const std::shared_ptr<DataSet>, int) override;
            const std::shared_ptr<VectorGroup<double>> get_feature(const std::string&) const override;
            const std::shared_ptr<VectorGroup<double>> get_means() const;
            const std::shared_ptr<VectorGroup<double>> get_stds() const;
            

    };


    

    

} // namespace cfade









    // class DiscreteStatistics: public FeatureGroupInterface{

        

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
