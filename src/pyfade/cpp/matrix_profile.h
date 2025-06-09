#pragma once


#include <cpp/dataset.h>

namespace cfade{

    class MatrixProfile : public FeatureGroupInterface{

        private:

            std::shared_ptr<VectorGroup<double>> matrix_profile;
            std::shared_ptr<VectorGroup<int>> index;
            std::shared_ptr<VectorGroup<double>> means;
            std::shared_ptr<VectorGroup<double>> stds;
            std::shared_ptr<VectorGroup<double>> QT;

            bool use_cuda=false;
            bool left_only=false;

            int exclusion_zone_size=0;
            

        public:

            inline static const std::string group_name = "MatrixProfile";
            inline static const std::vector<std::string> feature_names_string = {"matrix_profile","index"};

            inline const std::string& name() const override final { return group_name;}
            inline const std::vector<std::string>& feature_names() const override final { return feature_names_string;}

            MatrixProfile()=default;
            MatrixProfile(const std::shared_ptr<DataSet>, 
                const std::shared_ptr<VectorGroup<double>>, 
                const std::shared_ptr<VectorGroup<double>>);
            ~MatrixProfile()=default;

            void update(const std::shared_ptr<DataSet>, int) override;
            const std::shared_ptr<VectorGroup<double>> get_feature(const std::string&) const override;
    };

    class QTDataInterface{

        public:
            ~QTDataCUDA()=default;

    };


    class QTDataCUDA: public QTDataInterface{


        public:

            double* d_QT=nullptr;
            QTDataCUDA()=default;
            QTDataCUDA(const std::shared_ptr<DataSet>,
                                const std::shared_ptr<VectorGroup<double>>,
                                const std::shared_ptr<VectorGroup<double>>,
                                int,
                                int);
            ~QTDataCUDA();
    }
   



} // namespace cfade