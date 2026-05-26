#pragma once

#include <cfade/utils/utils.h>
#include <cfade/features/features.h>

#include <string>

namespace cfade{

    /**
     * @brief Class representing a matrix profile value feature.
     * 
     * This class is a CRTP (Curiously Recurring Template Pattern) implementation
     * of the IFeatureCRTP interface, specifically for matrix profile values.
     * 
     * It provides methods to access and modify the values in the matrix profile feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    class MatrixProfileValue: public IFeatureCRTP<MatrixProfileValue>{

        public:

            /**
             * @brief Type of the values stored in the feature vector.
             * 
             * This type is used to define the type of values that will be stored
             * in the MatrixProfileValue feature vector.
             */
            using value_type = double;

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            inline static const std::string feature_name = "matrix_profile_value";

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            static const std::string& static_name() { return feature_name;}

            /**
             * @brief Default constructor for MatrixProfileValue.
             * 
             * This constructor initializes the feature vector with zero rows.
             */
            MatrixProfileValue()
                : IFeatureCRTP<MatrixProfileValue>(0) {}

            /**
             * @brief Constructor for MatrixProfileValue.
             * 
             * This constructor initializes the feature vector with the specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */
            MatrixProfileValue(int rows)
                : IFeatureCRTP<MatrixProfileValue>(rows) {}

            /**
             * @brief Default destructor for MatrixProfileValue.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~MatrixProfileValue()=default;

            /**
             * @brief Access an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses IFeatureCRTP to retrieve
             * the appropriate vector type from the feature_vector.
             * 
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return const double The value at the specified row and column.
             */
            inline double at(int row, int col) const{
                return IFeatureCRTP<MatrixProfileValue>::template at<value_type>(row,col);
            }

            /**
             * @brief Get a reference to an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses IFeatureCRTP to retrieve
             * the appropriate vector type from the feature_vector.
             * 
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return double& A reference to the value at the specified row and column.
             */
            inline double& at(int row, int col){
                return IFeatureCRTP<MatrixProfileValue>::template at<value_type>(row,col);
            }

    };

    /**
     * @brief Class representing a matrix profile index feature.
     * 
     * This class is a CRTP (Curiously Recurring Template Pattern) implementation
     * of the IFeatureCRTP interface, specifically for matrix profile indices.
     * 
     * It provides methods to access and modify the indices in the matrix profile feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    class MatrixProfileIndex: public IFeatureCRTP<MatrixProfileIndex>{

        public:

            /**
             * @brief Type of the values stored in the feature vector.
             * 
             * This type is used to define the type of values that will be stored
             * in the MatrixProfileIndex feature vector.
             */
            using value_type = int;

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            using IFeatureCRTP<MatrixProfileIndex>::IFeatureCRTP;

            /**
             * @brief Name of the feature.
             * 
             * This static constant holds the name of the feature, which is used to identify it in the system.
             */
            inline static const std::string feature_name = "matrix_profile_index";

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            static const std::string& static_name() { return feature_name;}

            /**
             * @brief Default constructor for MatrixProfileIndex.
             * 
             * This constructor initializes the feature vector with zero rows.
             */
            MatrixProfileIndex()
                : IFeatureCRTP<MatrixProfileIndex>(0) {}

            /**
             * @brief Constructor for MatrixProfileIndex.
             * 
             * This constructor initializes the feature vector with the specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */
            MatrixProfileIndex(int rows)
                : IFeatureCRTP<MatrixProfileIndex>(rows) {}

            /**
             * @brief Default destructor for MatrixProfileIndex.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~MatrixProfileIndex()=default;

            /**
             * @brief Get the value at a specific row and column.
             * 
             * This method returns a const reference to the value at the specified
             * row and column in the feature vector.
             * 
             * @param row The row index.
             * @param col The column index.
             * @return const int& The value at the specified position.
             */
            inline int at(int row, int col) const{
                return IFeatureCRTP<MatrixProfileIndex>::template at<value_type>(row,col);
            }

            /**
             * @brief Get a reference to the value at a specific row and column.
             * 
             * This method returns a reference to the value at the specified
             * row and column in the feature vector, allowing modification of the value.
             * 
             * @param row The row index.
             * @param col The column index.
             * @return int& A reference to the value at the specified position.
             */
            inline int& at(int row, int col){
                return IFeatureCRTP<MatrixProfileIndex>::template at<value_type>(row,col);
            }

    };


} // namespace cfade