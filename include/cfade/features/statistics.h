#pragma once

#include <cfade/utils/utils.h>
#include <cfade/features/features.h>

#include <string>

namespace cfade
{

    /**
     * @brief Class representing a rolling mean feature.
     * 
     * This class is a CRTP (Curiously Recurring Template Pattern) implementation
     * of the IFeatureCRTP interface, specifically for rolling means.
     * 
     * It provides methods to access and modify the values in the rolling mean feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    class ContinuousMean: public IFeatureCRTP<ContinuousMean>{

        public:

            /**
             * @brief Type of the values stored in the feature vector.
             * 
             * This type is used to define the type of values that will be stored
             * in the ContinuousMean feature vector.
             */
            using value_type = double;

            /**
             * @brief Name of the feature.
             * 
             * This static constant holds the name of the feature, which is used to identify it in the system.
             */
            inline static const std::string feature_name = "mean";

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            static const std::string& static_name() { return feature_name;}

            /**
             * @brief Default constructor for ContinuousMean.
             * 
             * This constructor initializes the feature vector with zero rows.
             */
            ContinuousMean()
                : IFeatureCRTP<ContinuousMean>(0) {}

            /**
             * @brief Constructor for ContinuousMean.
             * 
             * This constructor initializes the feature vector with the specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */
            ContinuousMean(int rows)
                : IFeatureCRTP<ContinuousMean>(rows) {}

            /**
             * @brief Default destructor for ContinuousMean.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~ContinuousMean()=default;

            /**
             * @brief Get the value at a specific row and column.
             * 
             * This method returns a const reference to the value at the specified
             * row and column in the feature vector.
             * 
             * @param row The row index.
             * @param col The column index.
             * @return const double& The value at the specified position.
             */
            inline double at(int row, int col) const{
                return IFeatureCRTP<ContinuousMean>::template at<value_type>(row,col);
            }

            /**
             * @brief Get a reference to the value at a specific row and column.
             * 
             * This method returns a reference to the value at the specified
             * row and column in the feature vector, allowing modification of the value.
             * 
             * @param row The row index.
             * @param col The column index.
             * @return double& A reference to the value at the specified position.
             */
            inline double& at(int row, int col){
                return IFeatureCRTP<ContinuousMean>::template at<value_type>(row,col);
            }

    };



    /**
     * @brief Class representing a rolling standard deviation feature.
     * 
     * This class is a CRTP (Curiously Recurring Template Pattern) implementation
     * of the IFeatureCRTP interface, specifically for rolling standard deviations.
     * 
     * It provides methods to access and modify the values in the rolling standard deviation feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    class ContinuousStandardDeviation: public IFeatureCRTP<ContinuousStandardDeviation>{

        public:

            /**
             * @brief Type of the values stored in the feature vector.
             * 
             * This type is used to define the type of values that will be stored
             * in the ContinuousStandardDeviation feature vector.
             */
            using value_type = double;

            /**
             * @brief Name of the feature.
             * 
             * This static constant holds the name of the feature, which is used to identify it in the system.
             */
            inline static const std::string feature_name = "stddev";

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            static const std::string& static_name() { return feature_name;}
            
            /**
             * @brief Default constructor for ContinuousStandardDeviation.
             * 
             * This constructor initializes the feature vector with zero rows.
             */
            ContinuousStandardDeviation()
                : IFeatureCRTP<ContinuousStandardDeviation>(0) {}

            /**
             * @brief Constructor for ContinuousStandardDeviation.
             * 
             * This constructor initializes the feature vector with the specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */

            ContinuousStandardDeviation(int rows)
                : IFeatureCRTP<ContinuousStandardDeviation>(rows) {}

            /**
             * @brief Default destructor for ContinuousStandardDeviation.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~ContinuousStandardDeviation()=default;


            /**
             * @brief Get the value at a specific row and column.
             * 
             * This method returns a const reference to the value at the specified
             * row and column in the feature vector.
             * 
             * @param row The row index.
             * @param col The column index.
             * @return const double& The value at the specified position.
             */
            inline double at(int row, int col) const{
                return IFeatureCRTP<ContinuousStandardDeviation>::template at<value_type>(row,col);
            }


            /**
             * @brief Get a reference to the value at a specific row and column.
             * 
             * This method returns a reference to the value at the specified
             * row and column in the feature vector, allowing modification of the value.
             * 
             * @param row The row index.
             * @param col The column index.
             * @return double& A reference to the value at the specified position.
             */
            inline double& at(int row, int col){
                return IFeatureCRTP<ContinuousStandardDeviation>::template at<value_type>(row,col);
            }

    };

} // namespace cfade
