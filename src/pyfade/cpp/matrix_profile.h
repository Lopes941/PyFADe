#pragma once

#include <cpp/utils.h>
#include <cpp/dataset.h>
#include <cpp/statistics.h>
#include <cpp/window.h>
#include <cpp/features.h>

#include <vector>
#include <memory>
#include <string>

namespace cfade{

    // ===============
    // Features
    // ===============

    class MatrixProfileValue: public IFeatureCRTP<MatrixProfileValue>{

        public:

            using value_type = double;
            using IFeatureCRTP<MatrixProfileValue>::IFeatureCRTP;

            inline static const std::string feature_name = "matrix_profile_value";
            static const std::string& static_name() { return feature_name;}

            ~MatrixProfileValue()=default;

            inline double at(int row, int col) const{
                return IFeatureCRTP<MatrixProfileValue>::at<typename value_type>(row,col);
            }

            inline double& at(int row, int col){
                return IFeatureCRTP<MatrixProfileValue>::at<typename value_type>(row,col);
            }

    };

    class MatrixProfileIndex: public IFeatureCRTP<MatrixProfileIndex>{

        public:

            using value_type = int;
            using IFeatureCRTP<MatrixProfileIndex>::IFeatureCRTP;

            inline static const std::string feature_name = "matrix_profile_index";
            static const std::string& static_name() { return feature_name;}

            ~MatrixProfileIndex()=default;

            inline int at(int row, int col) const{
                return IFeatureCRTP<MatrixProfileIndex>::at<typename value_type>(row,col);
            }

            inline int& at(int row, int col){
                return IFeatureCRTP<MatrixProfileIndex>::at<typename value_type>(row,col);
            }

    };

    class KProfileValue: public IFeatureCRTP<KProfileValue>{

        public:

            using value_type = double;
            using IFeatureCRTP<KProfileValue>::IFeatureCRTP;

            inline static const std::string feature_name = "k_profile_value";
            static const std::string& static_name() { return feature_name;}

            ~KProfileValue()=default;

            inline double at(int row, int col) const{
                return IFeatureCRTP<KProfileValue>::at<typename value_type>(row,col);
            }

            inline double& at(int row, int col){
                return IFeatureCRTP<KProfileValue>::at<typename value_type>(row,col);
            }

    };

    class KProfileIndex: public IFeatureCRTP<KProfileIndex>{

        public:

            using value_type = int;
            using IFeatureCRTP<KProfileIndex>::IFeatureCRTP;

            inline static const std::string feature_name = "k_profile_index";
            static const std::string& static_name() { return feature_name;}

            ~KProfileIndex()=default;

            inline int at(int row, int col) const{
                return IFeatureCRTP<KProfileIndex>::at<typename value_type>(row,col);
            }

            inline int& at(int row, int col){
                return IFeatureCRTP<KProfileIndex>::at<typename value_type>(row,col);
            }

    };

    // ===============
    // Parameters
    // ===============

    class UseCudaParam: public FeatureGroupParameter<UseCudaParam>{

        public:

            using value_type = bool;

            inline static const std::string parameter_name = "use_cuda";
            static const std::string& static_name() { return parameter_name;}

            inline UseCudaParam(){set(false);}
            inline UseCudaParam(value_type val){set(val);}
            ~UseCudaParam()=default;

    };

    class LeftOnlyParam: public FeatureGroupParameter<LeftOnlyParam>{

        public:

            using value_type = bool;

            inline static const std::string parameter_name = "left_only";
            static const std::string& static_name() { return parameter_name;}

            inline LeftOnlyParam(){set(false);}
            inline LeftOnlyParam(value_type val){set(val);}
            ~LeftOnlyParam()=default;

    };

    class SkipStartParam: public FeatureGroupParameter<SkipStartParam>{

        public:

            using value_type = int;

            inline static const std::string parameter_name = "skip_start";
            static const std::string& static_name() { return parameter_name;}

            inline SkipStartParam(){set(-1);}
            inline SkipStartParam(value_type val){set(val);}
            ~SkipStartParam()=default;

    };

    class ExclusionZoneRatioParam: public FeatureGroupParameter<ExclusionZoneRatioParam>{

        public:

            using value_type = double;

            inline static const std::string parameter_name = "exclusion_zone_ratio";
            static const std::string& static_name() { return parameter_name;}

            inline ExclusionZoneRatioParam(){set(0.5);}
            inline ExclusionZoneRatioParam(value_type val){set(val);}
            ~ExclusionZoneRatioParam()=default;

    };

    // ===============
    // MatrixProfile Class
    // ===============

    class MatrixProfile : public IFeatureGroupCRTP<MatrixProfile>{

        private:
        
            inline static const std::string group_name = "MatrixProfile";
            inline static const std::vector<std::string> feature_names_string = {
                    MatrixProfileValue::static_name(),
                    MatrixProfileIndex::static_name()};
            inline static const std::vector<std::string> requirements_string = {ContinuousStatistics::static_name()};
            inline static const std::vector<std::string> parameters_string = {
                UseCudaParam::parameter_name,
                LeftOnlyParam::parameter_name,
                SkipStartParam::parameter_name,
                ExclusionZoneRatioParam::parameter_name,
            };

            std::shared_ptr<VectorGroup<double>> means;
            std::shared_ptr<VectorGroup<double>> stds;

            std::shared_ptr<MatrixProfileValue> matrix_profile;
            std::shared_ptr<MatrixProfileIndex> index;
            std::shared_ptr<VectorGroup<double>> QT;

            UseCudaParam use_cuda;
            LeftOnlyParam left_only; 
            SkipStartParam skip_start;
            ExclusionZoneRatioParam exclusion_zone_ratio;

        public:

            static const std::string& static_name() { return group_name;}
            static const std::vector<std::string>& static_feature_names() { return feature_names_string;}
            static const std::vector<std::string>& static_requirements() { return requirements_string;}
            static const std::vector<std::string>& static_parameters() { return parameters_string;}

            void set_parameter(const std::string&, VariantTypes) override;

            MatrixProfile()=default;
            MatrixProfile(const std::shared_ptr<DataSet>, 
                const std::vector<std::shared_ptr<IFeatureGroup>>);
            ~MatrixProfile()=default;

            void update(const std::shared_ptr<DataSet>, int) override;
    };

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
            KProfile(const std::shared_ptr<DataSet>, 
                const std::vector<std::shared_ptr<IFeatureGroup>>);
            ~KProfile()=default;

            void update(const std::shared_ptr<DataSet>, int) override;
    };

    // ===============
    // KProfile Class
    // ===============
    
    // ===============
    // QTData Class
    // ===============

    class QTDataInterface{

        public:
            double* d_QT=nullptr;
            ~QTDataInterface()=default;
            

    };


    class QTDataCUDA: public QTDataInterface{


        public:
            QTDataCUDA()=default;
            QTDataCUDA(const std::shared_ptr<DataSet>,
                                const std::shared_ptr<VectorGroup<double>>,
                                const std::shared_ptr<VectorGroup<double>>,
                                int,
                                int);
            ~QTDataCUDA();
    };
   



} // namespace cfade