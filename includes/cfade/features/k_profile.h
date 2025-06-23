#pragma once

#include <cfade/utils/utils.h>
#include <cfade/features/features.h>

#include <string>

namespace cfade
{

    /**
     * @brief Class representing a K-Profile value feature.
     * 
     * This class is a CRTP (Curiously Recurring Template Pattern) implementation
     * of the IFeatureCRTP interface, specifically for K-Profile values.
     * 
     * It provides methods to access and modify the values in the K-Profile feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    class KProfileValue: public IFeatureCRTP<KProfileValue>{

        public:

            /**
             * @brief Type of the values stored in the feature vector.
             * 
             * This type is used to define the type of values that will be stored
             * in the KProfileValue feature vector.
             */
            using value_type = double;

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            inline static const std::string feature_name = "k_profile_value";

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            static const std::string& static_name() { return feature_name;}

            /**
             * @brief Default constructor for KProfileValue.
             * 
             * This constructor initializes the feature vector with zero rows.
             */
            KProfileValue()
                : IFeatureCRTP<KProfileValue>(0) {}

            /**
             * @brief Constructor for KProfileValue.
             * 
             * This constructor initializes the feature vector with the specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */
            KProfileValue(int rows)
                : IFeatureCRTP<KProfileValue>(rows) {}

            /**
             * @brief Default destructor for KProfileValue.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~KProfileValue()=default;

            /**
             * @brief Access an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses IFeatureCRTP's at method
             * to retrieve the appropriate value type.
             * 
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return const double The value at the specified row and column.
             */
            inline double at(int row, int col) const{
                return IFeatureCRTP<KProfileValue>::at<typename value_type>(row,col);
            }

            /**
             * @brief Get a reference to an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses IFeatureCRTP's at method
             * to retrieve the appropriate value type.
             * 
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return double& A reference to the value at the specified row and column.
             */
            inline double& at(int row, int col){
                return IFeatureCRTP<KProfileValue>::at<typename value_type>(row,col);
            }

    };

    /**
     * @brief Class representing a K-Profile index feature.
     * 
     * This class is a CRTP (Curiously Recurring Template Pattern) implementation
     * of the IFeatureCRTP interface, specifically for K-Profile indices.
     * 
     * It provides methods to access and modify the indices in the K-Profile feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    class KProfileIndex: public IFeatureCRTP<KProfileIndex>{

        public:

            /**
             * @brief Type of the values stored in the feature vector.
             * 
             * This type is used to define the type of values that will be stored
             * in the KProfileIndex feature vector.
             */
            using value_type = int;

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            inline static const std::string feature_name = "k_profile_index";

            /**
             * @brief Get the name of the feature.
             * 
             * This static method returns the name of the feature, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature.
             */
            static const std::string& static_name() { return feature_name;}

            /**
             * @brief Default constructor for KProfileIndex.
             * 
             * This constructor initializes the feature vector with zero rows.
             */
            KProfileIndex()
                : IFeatureCRTP<KProfileIndex>(0) {}

            /**
             * @brief Constructor for KProfileIndex.
             * 
             * This constructor initializes the feature vector with the specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */
            KProfileIndex(int rows)
                : IFeatureCRTP<KProfileIndex>(rows) {}

            /**
             * @brief Default destructor for KProfileIndex.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~KProfileIndex()=default;

            /**
             * @brief Access an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses IFeatureCRTP's at method
             * to retrieve the appropriate value type.
             * 
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return const int The value at the specified row and column.
             */
            inline int at(int row, int col) const{
                return IFeatureCRTP<KProfileIndex>::at<typename value_type>(row,col);
            }

            /**
             * @brief Get a reference to an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses IFeatureCRTP's at method
             * to retrieve the appropriate value type.
             * 
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return int& A reference to the value at the specified row and column.
             */
            inline int& at(int row, int col){
                return IFeatureCRTP<KProfileIndex>::at<typename value_type>(row,col);
            }

    };
    
} // namespace cfade
