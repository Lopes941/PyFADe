#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>
#include <cfade/features/features.h>

#include <string>
#include <variant>
#include <unordered_map>
#include <memory>
#include <vector>
#include <stdexcept>


namespace cfade{

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

            std::unordered_map<std::string, std::shared_ptr<IFeature>> feature_map; /**< Map of feature names to their corresponding IFeature objects. */

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
             * @param observed_dataset The dataset containing the observed data to update the features.
             * @param window_size The size of the window to consider for the updates.
             */
            virtual void update(const std::shared_ptr<DataSet>& observed_dataset, int window_size)=0;

        
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
                    feature_map[feature_name] = feature_builder_map[feature_name](0);
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
                    feature_map[feature_name] = feature_builder_map[feature_name](dimension);
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
    
    /**
     * @brief Interface for feature group parameters.
     * 
     * This interface defines the methods that all feature group parameters must implement.
     */
    class IFeatureGroupParameter{

        public:

            /**
             * @brief Default destructor.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~IFeatureGroupParameter()=default;

            /**
             * @brief Get the name of the parameter.
             * 
             * This method returns the name of the parameter as a string.
             * 
             * @return const std::string& The name of the parameter.
             */
            virtual const std::string& name() const = 0;

            /**
             * @brief Set the value of the parameter.
             * 
             * This method sets the value of the parameter using a variant type.
             * 
             * @param value The value to set.
             */
            virtual void set(VariantTypes)=0;
            
    };


    /**
     * @brief Template class for feature group parameters.
     * 
     * This class implements the IFeatureGroupParameter interface and provides a way to set and get
     * the value of a parameter. It uses CRTP (Curiously Recurring Template Pattern) to allow derived
     * classes to specify their own value type.
     * 
     * @tparam Derived The derived class that implements the feature group parameter.
     */
    template <typename Derived>
    class FeatureGroupParameter: public IFeatureGroupParameter{

        private:

            VariantTypes value; /**< The value of the parameter, stored as a variant type. */

        public:

            /**
             * @brief Default constructor.
             * 
             * This constructor initializes the parameter with a default value.
             */
            ~FeatureGroupParameter()=default;

            /**
             * @brief Get the name of the parameter.
             * 
             * This method returns the name of the parameter as a string.
             * 
             * @return const std::string& The name of the parameter.
             */
            inline const std::string& name() const override final{
                return Derived::static_name();
            }

            /**
             * @brief Set the value of the parameter.
             * 
             * This method sets the value of the parameter using a variant type.
             * It converts the variant to the appropriate type defined in the derived class.
             * 
             * @param val The value to set, as a variant type.
             */
            inline void set(VariantTypes val) override final{
                value = std::visit(
                    [](auto&& arg) -> typename Derived::value_type {
                        return static_cast<typename Derived::value_type>(arg);
                    },val);
            }

            /**
             * @brief Get the value of the parameter.
             * 
             * This method returns the value of the parameter as a variant type.
             * 
             * @return VariantTypes The value of the parameter.
             */
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
