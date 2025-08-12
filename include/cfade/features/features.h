#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <typeindex>
#include <stdexcept>
#include <functional>


namespace cfade{

    /**
     * @brief Interface for a feature.
     * 
     * This interface defines the basic structure for a feature in the system.
     * It provides methods to retrieve the name of the feature and to get the
     * feature vector as a VariantVec.
     * 
     * The feature vector can be of different types, such as a vector of integers
     * or a vector of doubles, allowing for flexibility in the types of features
     * that can be implemented.
     */
    class IFeature{

        public:

            /**
             * @brief Get the name of the feature.
             * 
             * This method should return the name of the feature as a string.
             * It is expected that each derived class will implement this method
             * to return a unique name for the feature.
             * 
             * @return const std::string& Name of the feature.
             */
            virtual const std::string& name() const = 0;

            /**
             * @brief Get the feature vector.
             * 
             * This method should return the feature vector as a VariantVec.
             * The VariantVec can hold different types of vectors, such as
             * std::shared_ptr<VectorGroup<int>> or std::shared_ptr<VectorGroup<double>>.
             * 
             * @return VariantVec The feature vector.
             */
            virtual VariantVec get() =0;

    };

    /**
     * @brief CRTP (Curiously Recurring Template Pattern) implementation of IFeature.
     * 
     * This class implements the IFeature interface using the CRTP pattern.
     * It allows derived classes to define their own feature types while
     * providing a common interface for accessing the feature vector.
     * 
     * The Derived class must implement a static_name() method to return the name
     * of the feature type and a value_type to define the type of values stored in
     * the feature vector.
     * 
     * @tparam Derived The derived class that implements the feature.
     */
    template <typename Derived>
    class IFeatureCRTP : public IFeature{

        private:
            
            VariantVec feature_vector; /**< The feature vector, which can hold different types of vectors. */

        public:

            /**
             * @brief Get the name of the feature.
             * 
             * This method returns the name of the feature by calling the static_name()
             * method of the Derived class. It is expected that each derived class will
             * implement this method to return a unique name for the feature.
             *  
             * @return const std::string& Name of the feature.
             */
            inline const std::string& name() const final{
                return Derived::static_name();
            } 

            /**
             * @brief Get the length of the feature vector.
             * 
             * This method returns the number of columns in the feature vector.
             * It uses std::visit to handle different types of vectors stored in
             * the VariantVec feature_vector.
             * 
             * @return int The number of columns in the feature vector.
             */
            inline int get_length() const{
                return std::visit([](auto &vec) -> int {
                    return vec->cols;
                },feature_vector);
            }

            /**
             * @brief Increase the number of columns in the feature vector.
             * 
             * This method increases the number of columns in the feature vector
             * by a specified amount. It uses std::visit to handle different types
             * of vectors stored in the VariantVec feature_vector.
             * 
             * @param added_size The number of columns to be added.
             */
            inline void increase_cols(const int added_size){
                std::visit([added_size](auto &vec){
                    vec->increase_cols(added_size);
                },feature_vector);
            }

            /**
             * @brief Access an element in the feature vector.
             * 
             * This method allows access to an element in the feature vector
             * at a specified row and column. It uses std::get to retrieve the
             * appropriate vector type from the VariantVec feature_vector.
             * 
             * @tparam T The type of the element to be accessed.
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return const T The value at the specified row and column.
             */
            template<typename T>
            inline const T at(int row, int col) const{
                using Type = std::shared_ptr<VectorGroup<T>>;
                return std::get<Type>(feature_vector)->at(row,col);
            }

            /**
             * @brief Access an element in the feature vector (mutable).
             * 
             * This method allows mutable access to an element in the feature vector
             * at a specified row and column. It uses std::get to retrieve the
             * appropriate vector type from the VariantVec feature_vector.
             * 
             * @tparam T The type of the element to be accessed.
             * @param row The row index of the element.
             * @param col The column index of the element.
             * @return T& Reference to the value at the specified row and column.
             */
            template<typename T>
            inline T& at(int row, int col){
                using Type = std::shared_ptr<VectorGroup<T>>;
                return std::get<Type>(feature_vector)->at(row,col);
            }

            /**
             * @brief Get the feature vector.
             * 
             * This method returns the feature vector as a VariantVec.
             * It is expected that each derived class will implement this
             * method to return the appropriate feature vector type.
             * 
             * @return VariantVec The feature vector.
             */
            inline VariantVec get() final{
                return feature_vector;
            }

            /**
             * @brief Default constructor for IFeatureCRTP.
             * 
             * This constructor initializes the feature_vector with a
             * VectorGroup of the appropriate type defined in the Derived class.
             */
            inline IFeatureCRTP(){
                feature_vector = std::make_shared<VectorGroup<typename Derived::value_type>>
                    (VectorGroup<typename Derived::value_type>(0));
            }
            
            /**
             * @brief Constructor for IFeatureCRTP with specified number of rows.
             * 
             * This constructor initializes the feature_vector with a
             * VectorGroup of the appropriate type defined in the Derived class
             * and a specified number of rows.
             * 
             * @param rows The number of rows in the feature vector.
             */
            inline IFeatureCRTP(int rows){
                feature_vector = std::make_shared<VectorGroup<typename Derived::value_type>>
                    (VectorGroup<typename Derived::value_type>(rows));
            }

            /**
             * @brief Default destructor for IFeatureCRTP.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~IFeatureCRTP()=default;
    };


    /**
     * @brief Map of feature names to their Factories.
     * 
     * This map associates each feature name with its corresponding factory.
     * It allows for dynamic instantiation of IFeature Derived classes.
     * 
     * 
     * @param feature_name The name of the feature.
     * @return FeatureFactory The corresponding factory for the IFeature with feature_name.
     */
    // std::unordered_map<std::string, std::type_index> feature_type_map;
    using FeatureFactory = std::function<std::shared_ptr<IFeature>(int row)>;
    extern std::unordered_map<std::string, FeatureFactory> feature_builder_map;

}  // namespace cfade