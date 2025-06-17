#pragma once

#include <cpp/utils.h> 
#include <cpp/window.h>

#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>


//  Problema na ordenação das coisas, TODO

namespace cfade{

inline std::unordered_map<std::string, std::type_index> feature_type_map = {
    {ContinuousMean::static_name(), std::type_index(typeid(ContinuousMean::value_type))}
};


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
            inline const std::string& name() const override{
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
            inline VariantVec get() override{
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
     * @brief Interface for a feature group.
     * 
     * This interface defines the basic structure for a group of features in the system.
     * It provides methods to retrieve the name of the feature group, the names of the features,
     * and to update the features based on a dataset.
     * 
     * The feature group can contain multiple features, which can be accessed by their names.
     */
    class IFeatureGroup{

        protected:

            std::map<std::string, std::shared_ptr<IFeature>> feature_map; /**< Map of feature names to their corresponding IFeature objects. */

        public:
        
            /**
             * @brief Default destructor for IFeatureGroup.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~IFeatureGroup()=default;

            /**
             * @brief Get the name of the feature group.
             * 
             * This method should return the name of the feature group as a string.
             * It is expected that each derived class will implement this method
             * to return a unique name for the feature group.
             * 
             * @return const std::string& Name of the feature group.
             */
            virtual const std::string& name() const = 0;

            /**
             * @brief Get the names of the features in the group.
             * 
             * This method should return a vector of strings containing the names
             * of the features in the group. It is expected that each derived class
             * will implement this method to return the appropriate feature names.
             * 
             * @return const std::vector<std::string>& Names of the features in the group.
             */
            virtual const std::vector<std::string>& feature_names() const = 0;

            /**
             * @brief Get the requirements of the feature group.
             * 
             * This method should return a vector of strings containing the names
             * of the required feature groups for this feature group. It is expected
             * that each derived class will implement this method to return the
             * appropriate requirements.
             * 
             * @return const std::vector<std::string>& Requirements of the feature group.
             */
            virtual const std::vector<std::string>& requirements() const = 0;

            /**
             * @brief Get the parameters of the feature group.
             * 
             * This method should return a vector of strings containing the names
             * of the parameters for this feature group. It is expected that each
             * derived class will implement this method to return the appropriate
             * parameters.
             * 
             * @return const std::vector<std::string>& Parameters of the feature group.
             */
            virtual const std::vector<std::string>& parameters() const = 0;
            
            /**
             * @brief Set a parameter for the feature group.
             * 
             * This method should set the value of a specified parameter for the feature group.
             * If the parameter does not exist, it should throw an exception.
             * 
             * @param key The name of the parameter to be set.
             * @param val The value to be set for the parameter.
             */
            inline virtual void set_parameter(const std::string& key, VariantTypes val){
                throw std::invalid_argument(name() + "' has no parameters, but '" + key + "' was provided.");
            }

            /**
             * @brief Get a feature by its name.
             * 
             * This method retrieves the feature vector for a specified feature name.
             * It uses the feature_map to find the corresponding IFeature object and
             * returns its feature vector.
             * 
             * @param name The name of the feature to be retrieved.
             * @return VariantVec The feature vector for the specified feature name.
             */
            inline const VariantVec get_feature(const std::string& name) const{
                return feature_map.at(name)->get();
            }

            /**
             * @brief Update the feature group based on a dataset.
             * 
             * This method should update the features in the group based on the provided
             * dataset and a specified window size. It is expected that each derived class
             * will implement this method to perform the necessary updates.
             * 
             * @param dataset The dataset used for updating the features.
             * @param window_size The size of the window for updating the features.
             */
            virtual void update(const std::shared_ptr<DataSet> dataset, int window_size)=0;

        
    };


    /**
     * @brief CRTP (Curiously Recurring Template Pattern) implementation of IFeatureGroup.
     * 
     * This class implements the IFeatureGroup interface using the CRTP pattern.
     * It allows derived classes to define their own feature groups while
     * providing a common interface for accessing the feature names and requirements.
     * 
     * The Derived class must implement static_name(), static_feature_names(),
     * static_requirements(), and static_parameters() methods to return the
     * appropriate values for the feature group.
     * 
     * @tparam Derived The derived class that implements the feature group.
     */
    template <typename Derived>
    class IFeatureGroupCRTP : public IFeatureGroup{

        public:

        
            /**
             * @brief Default constructor for IFeatureGroupCRTP.
             * 
             * This constructor initializes the feature_map with the features defined in the Derived class.
             */
            ~IFeatureGroupCRTP()=default;

            /**
             * @brief Constructor for IFeatureGroupCRTP.
             * 
             * This constructor initializes the feature_map with the features defined in the Derived class.
             * It iterates over the static_feature_names() of the Derived class and creates
             * a shared pointer for each feature type, storing them in the feature_map.
             * @param Derived The derived class that implements the feature group.
             */
            inline IFeatureGroupCRTP(){
                for(const auto& feature_name : Derived::static_feature_names()){
                    feature_map[feature_name] = std::make_shared<feature_type_map[feature_name]>();
                }
            }

            /**
             * @brief Constructor for IFeatureGroupCRTP with specified dimension.
             * 
             * This constructor initializes the feature_map with the features defined in the Derived class
             * and a specified dimension. It iterates over the static_feature_names() of the Derived class
             * and creates a shared pointer for each feature type, storing them in the feature_map.
             * 
             * @param dimension The dimension of the features to be created.
             */
            inline IFeatureGroupCRTP(int dimension){
                for(const auto& feature_name : Derived::static_feature_names()){
                    feature_map[feature_name] = std::make_shared<feature_type_map[feature_name]>(dimension);
                }
            }

            /**
             * @brief Get the name of the feature group.
             * 
             * This method returns the name of the feature group by calling the static_name()
             * method of the Derived class. It is expected that each derived class will
             * implement this method to return a unique name for the feature group.
             * 
             * @return const std::string& Name of the feature group.
             */
            inline const std::string& name() const override{
                return Derived::static_name();
            }

            /**
             * @brief Get the names of the features in the group.
             * 
             * This method returns a vector of strings containing the names
             * of the features in the group by calling the static_feature_names()
             * method of the Derived class. It is expected that each derived class
             * will implement this method to return the appropriate feature names.
             * 
             * @return const std::vector<std::string>& Names of the features in the group.
             */
            inline const std::vector<std::string>& feature_names() const override{
                return Derived::static_feature_names();
            }

            /**
             * @brief Get the requirements of the feature group.
             * 
             * This method returns a vector of strings containing the names
             * of the required feature groups for this feature group by calling
             * the static_requirements() method of the Derived class. It is expected
             * that each derived class will implement this method to return the
             * appropriate requirements.
             * 
             * @return const std::vector<std::string>& Requirements of the feature group.
             */
            inline const std::vector<std::string>& requirements() const override{
                return Derived::static_requirements();
            }

            /**
             * @brief Get the parameters of the feature group.
             * 
             * This method returns a vector of strings containing the names
             * of the parameters for this feature group by calling the static_parameters()
             * method of the Derived class. It is expected that each derived class
             * will implement this method to return the appropriate parameters.
             * 
             * @return const std::vector<std::string>& Parameters of the feature group.
             */
            inline const std::vector<std::string>& parameters() const override{
                return Derived::static_parameters();
            }

            /**
             * @brief Set a parameter for the feature group.
             * 
             * This method sets the value of a specified parameter for the feature group.
             * If the parameter does not exist, it throws an exception.
             * 
             * @param key The name of the parameter to be set.
             * @param val The value to be set for the parameter.
             */

            // inline void set_parameter(const std::string& key, VariantTypes val) override{
            //     auto it = feature_map.find(key);
            //     if (it != feature_map.end()) {
            //         it->second->set(val);
            //     } else {
            //         throw std::invalid_argument(name() + " has no '" + key + "' parameter.");
            //     }
            // }

            /**
             * @brief Update the feature group based on a dataset.
             * 
             * This method updates the features in the group based on the provided
             * dataset and a specified window size. It is expected that each derived class
             * will implement this method to perform the necessary updates.
             * 
             * @param dataset The dataset used for updating the features.
             * @param window_size The size of the window for updating the features.
             */

            // inline void update(const std::shared_ptr<DataSet> dataset, int window_size) override{
            //     Derived (dataset,window_size);

            // }
            

            /**
             * @brief Get a feature by its name.
             * 
             * This method retrieves the feature vector for a specified feature name.
             * It uses the feature_map to find the corresponding IFeature object and
             * returns its feature vector.
             * 
             * @param name The name of the feature to be retrieved.
             * @return VariantVec The feature vector for the specified feature name.
             */
            inline const VariantVec get_feature(const std::string& name) const{
                return feature_map.at(name)->get();
            }

    };    

    

}  // namespace cfade