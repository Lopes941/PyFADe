#pragma once

#include <cfade/utils/utils.h>
#include <cfade/dataset/dataset.h>
#include <cfade/groups/groups.h>
#include <cfade/groups/statistics.h>
#include <cfade/features/features.h>
#include <cfade/features/matrix_profile.h>

#include <vector>
#include <memory>
#include <string>

namespace cfade{

    /**
     * @brief Parameter for using CUDA acceleration.
     * 
     * This parameter enables or disables the use of CUDA for acceleration.
     * It is a boolean parameter that can be set to true or false.
     * This class inherits from FeatureGroupParameter and implements the
     * necessary methods for a feature group parameter.
     * 
     * @tparam Derived The derived class that implements the feature group parameter.
     */
    class UseCudaParam: public FeatureGroupParameter<UseCudaParam>{

        public:

            /**
             * @brief Type of the value for the UseCudaParam.
             * 
             * This type is a boolean indicating whether to use CUDA or not.
             */
            using value_type = bool; 

            /**
             * @brief Name of the parameter.
             * 
             * This static constant holds the name of the parameter, which is used to identify it in the system.
             */
            inline static const std::string parameter_name = "use_cuda";

            /**
             * @brief Get the name of the parameter.
             * 
             * This static method returns the name of the parameter, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the parameter.
             */
            static const std::string& static_name() { return parameter_name;}

            /**
             * @brief Default constructor for UseCudaParam.
             * 
             * This constructor initializes the parameter with a default value of false.
             */
            inline UseCudaParam(){set(false);}

            /**
             * @brief Constructor for UseCudaParam with a specified value.
             * 
             * This constructor initializes the parameter with the specified value.
             * 
             * @param val The value to set for the parameter, as a boolean.
             */
            inline UseCudaParam(value_type val){set(val);}

            /**
             * @brief Default destructor for UseCudaParam.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~UseCudaParam()=default;

    };

    /**
     * @brief Parameter for left-only computation in matrix profile.
     * 
     * This parameter indicates whether to compute only the left side of the matrix profile.
     * It is a boolean parameter that can be set to true or false.
     * This class inherits from FeatureGroupParameter and implements the
     * necessary methods for a feature group parameter.
     * 
     * @tparam Derived The derived class that implements the feature group parameter.
     */
    class LeftOnlyParam: public FeatureGroupParameter<LeftOnlyParam>{

        public:

            /**
             * @brief Type of the value for the LeftOnlyParam.
             * 
             * This type is a boolean indicating whether to compute only the left side of the matrix profile.
             */
            using value_type = bool;

            /**
             * @brief Name of the parameter.
             * 
             * This static constant holds the name of the parameter, which is used to identify it in the system.
             */
            inline static const std::string parameter_name = "left_only";

            /**
             * @brief Get the name of the parameter.
             * 
             * This static method returns the name of the parameter, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the parameter.
             */
            static const std::string& static_name() { return parameter_name;}

            /**
             * @brief Default constructor for LeftOnlyParam.
             * 
             * This constructor initializes the parameter with a default value of false.
             */
            inline LeftOnlyParam(){set(false);}

            /**
             * @brief Constructor for LeftOnlyParam with a specified value.
             * 
             * This constructor initializes the parameter with the specified value.
             * 
             * @param val The value to set for the parameter, as a boolean.
             */
            inline LeftOnlyParam(value_type val){set(val);}

            /**
             * @brief Default destructor for LeftOnlyParam.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~LeftOnlyParam()=default;

    };

    /**
     * @brief Parameter for skipping the start of the matrix profile computation.
     * 
     * This parameter allows for skipping a specified number of elements at the start of the matrix profile computation.
     * It is an integer parameter that can be set to a negative value to indicate no skip.
     * This class inherits from FeatureGroupParameter and implements the
     * necessary methods for a feature group parameter.
     * 
     * @tparam Derived The derived class that implements the feature group parameter.
     */
    class SkipStartParam: public FeatureGroupParameter<SkipStartParam>{

        public:

            /**
             * @brief Type of the value for the SkipStartParam.
             * 
             * This type is an integer indicating the number of elements to skip at the start of the matrix profile computation.
             */
            using value_type = int;

            /**
             * @brief Name of the parameter.
             * 
             * This static constant holds the name of the parameter, which is used to identify it in the system.
             */
            inline static const std::string parameter_name = "skip_start";

            /**
             * @brief Get the name of the parameter.
             * 
             * This static method returns the name of the parameter, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the parameter.
             */
            static const std::string& static_name() { return parameter_name;}

            /**
             * @brief Default constructor for SkipStartParam.
             * 
             * This constructor initializes the parameter with a default value of -1, a default skip of one window.
             */
            inline SkipStartParam(){set(-1);}

            /**
             * @brief Constructor for SkipStartParam with a specified value.
             * 
             * This constructor initializes the parameter with the specified value.
             * 
             * @param val The value to set for the parameter, as an integer.
             */
            inline SkipStartParam(value_type val){set(val);}

            /**
             * @brief Default destructor for SkipStartParam.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~SkipStartParam()=default;

    };

    /**
     * @brief Parameter for the exclusion zone ratio in matrix profile.
     * 
     * This parameter specifies the ratio of the exclusion zone to the window size in the matrix profile computation. The exclusion zone is then calculated as a ratio of the window size.
     * The exclusion zone is the area around a found motif where no other motifs can be found to avoid trivial matches.
     * It is a double parameter that can be set to any positive value.
     * This class inherits from FeatureGroupParameter and implements the
     * necessary methods for a feature group parameter.
     * 
     * @tparam Derived The derived class that implements the feature group parameter.
     */
    class ExclusionZoneRatioParam: public FeatureGroupParameter<ExclusionZoneRatioParam>{

        public:

            /**
             * @brief Type of the value for the ExclusionZoneRatioParam.
             * 
             * This type is a double indicating the ratio of the exclusion zone to the window size.
             */
            using value_type = double;

            /**
             * @brief Name of the parameter.
             * 
             * This static constant holds the name of the parameter, which is used to identify it in the system.
             */
            inline static const std::string parameter_name = "exclusion_zone_ratio";

            /**
             * @brief Get the name of the parameter.
             * 
             * This static method returns the name of the parameter, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the parameter.
             */
            static const std::string& static_name() { return parameter_name;}


            /**
             * @brief Default constructor for ExclusionZoneRatioParam.
             * 
             * This constructor initializes the parameter with a default value of 0.5.
             */
            inline ExclusionZoneRatioParam(){set(0.5);}

            /**
             * @brief Constructor for ExclusionZoneRatioParam with a specified value.
             * 
             * This constructor initializes the parameter with the specified value.
             * 
             * @param val The value to set for the parameter, as a double.
             */
            inline ExclusionZoneRatioParam(value_type val){set(val);}

            /**
             * @brief Default destructor for ExclusionZoneRatioParam.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~ExclusionZoneRatioParam()=default;

    };

    /**
     * @brief Parameter for the low quantile threshold in matrix profile.
     * 
     * This parameter specifies the quantile to be defined as the 0 of the matrix profile.
     * This is a parameter to be added in very noisy signals, so only peak Matrix Profile values are returned.
     * 
     * @tparam Derived The derived class that implements the feature group parameter.
     */
    class QuantileParam: public FeatureGroupParameter<QuantileParam>{

        public:

            /**
             * @brief Type of the value for the ExclusionZoneRatioParam.
             * 
             * This type is a double indicating the ratio of the exclusion zone to the window size.
             */
            using value_type = double;

            /**
             * @brief Name of the parameter.
             * 
             * This static constant holds the name of the parameter, which is used to identify it in the system.
             */
            inline static const std::string parameter_name = "quantile_threshold";

            /**
             * @brief Get the name of the parameter.
             * 
             * This static method returns the name of the parameter, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the parameter.
             */
            static const std::string& static_name() { return parameter_name;}


            /**
             * @brief Default constructor for ExclusionZoneRatioParam.
             * 
             * This constructor initializes the parameter with a default value of 0.5.
             */
            inline QuantileParam(){set(0.);}

            /**
             * @brief Constructor for ExclusionZoneRatioParam with a specified value.
             * 
             * This constructor initializes the parameter with the specified value.
             * 
             * @param val The value to set for the parameter, as a double.
             */
            inline QuantileParam(value_type val){set(val);}

            /**
             * @brief Default destructor for ExclusionZoneRatioParam.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~QuantileParam()=default;

    };



    /**
     * @brief Matrix profile feature group.
     * 
     * This class represents the matrix profile feature group, which includes various parameters
     * and methods for computing the matrix profile.
     * The matrix profile is a data structure that captures the similarity between subsequences of a time series.
     * It is used for tasks such as motif discovery, anomaly detection, and similarity search.
     * This class inherits from IFeatureGroupCRTP and implements the necessary methods for a feature group.
     * 
     * @tparam Derived The derived class that implements the feature group.
     */
    class MatrixProfile : public IFeatureGroupCRTP<MatrixProfile>{

        private:
        
            /**
             * @brief Group name for the MatrixProfile feature group.
             * 
             * This static constant holds the name of the feature group, which is used to identify it in the system.
             */
            inline static const std::string group_name = "MatrixProfile";

            /**
             * @brief Feature names for the MatrixProfile feature group.
             * 
             * This static constant holds the names of the features in the group, which are used to identify them in the system.
             */
            inline static const std::vector<std::string> feature_names_string = {
                    MatrixProfileValue::static_name(),
                    MatrixProfileIndex::static_name()};

            /**
             * @brief Requirements for the MatrixProfile feature group.
             * 
             * This static constant holds the names of the required feature groups for this feature group.
             * Currently, it requires the ContinuousStatistics feature group.
             */
            inline static const std::vector<std::string> requirements_string = {ContinuousStatistics::static_name()};

            /**
             * @brief Parameters for the MatrixProfile feature group.
             * 
             * This static constant holds the names of the parameters for this feature group.
             */
            inline static const std::vector<std::string> parameters_string = {
                UseCudaParam::parameter_name,
                LeftOnlyParam::parameter_name,
                SkipStartParam::parameter_name,
                ExclusionZoneRatioParam::parameter_name,
                QuantileParam::parameter_name,
            };


            std::shared_ptr<VectorGroup<double>> means; /**< Mean values of the matrix profile */
            std::shared_ptr<VectorGroup<double>> stds; /**< Standard deviation values of the matrix profile */
            std::shared_ptr<VectorGroup<double>> raw_matrix_profile; /**< Matrix profile values */

            std::shared_ptr<MatrixProfileValue> matrix_profile; /**< Matrix profile values after quantile evaluation */
            std::shared_ptr<MatrixProfileIndex> index; /**< Indices of the matrix profile */
            std::shared_ptr<VectorGroup<double>> QT; /**< QT values for the matrix profile */

            UseCudaParam use_cuda; /**< Use CUDA for computation */
            LeftOnlyParam left_only; /**< Use left-only matching */
            SkipStartParam skip_start; /**< Skip start of the time series */
            ExclusionZoneRatioParam exclusion_zone_ratio; /**< Exclusion zone ratio */
            QuantileParam quantile_param; /**< Excluded quantile number */

        public:

            /**
             * @brief Get the name of the feature group.
             * 
             * This static method returns the name of the feature group, which is used to identify it in the system.
             * 
             * @return const std::string& The name of the feature group.
             */
            static const std::string& static_name() { return group_name;}

            /**
             * @brief Get the names of the features in the group.
             * 
             * This static method returns a vector of strings containing the names
             * of the features in the group. It is expected that each derived class
             * will implement this method to return the appropriate feature names.
             * 
             * @return const std::vector<std::string>& Names of the features in the group.
             */
            static const std::vector<std::string>& static_feature_names() { return feature_names_string;}

            /**
             * @brief Get the requirements of the feature group.
             * 
             * This static method returns a vector of strings containing the names
             * of the required feature groups for this feature group. It is expected
             * that each derived class will implement this method to return the
             * appropriate requirements.
             * 
             * @return const std::vector<std::string>& Requirements of the feature group.
             */
            static const std::vector<std::string>& static_requirements() { return requirements_string;}

            /**
             * @brief Get the parameters of the feature group.
             * 
             * This static method returns a vector of strings containing the names
             * of the parameters for this feature group. It is expected that each
             * derived class will implement this method to return the appropriate
             * parameters.
             * 
             * @return const std::vector<std::string>& Parameters of the feature group.
             */
            static const std::vector<std::string>& static_parameters() { return parameters_string;}

            /**
             * @brief Set a parameter for the feature group.
             * 
             * This method sets the value of a specified parameter for the feature group.
             * If the parameter does not exist, it throws an exception.
             * 
             * @param name The name of the parameter to set.
             * @param val The value to set for the parameter, as a VariantTypes.
             */
            void set_parameter(const std::string&, VariantTypes) override;

            /**
             * @brief Default constructor for MatrixProfile.
             * 
             * This constructor initializes the feature group with zero rows.
             */
            MatrixProfile()=default;

            /**
             * @brief Parameterized constructor for MatrixProfile.
             * 
             * This constructor initializes the feature group with the given dataset and feature groups.
             * 
             * @param observed_dataset The dataset to use for the feature group.
             * @param requirements The required feature groups for this feature group.
             */
            MatrixProfile(const std::shared_ptr<DataSet>& observed_dataset, 
                const std::vector<std::shared_ptr<IFeatureGroup>>& requirements);

            /**
             * @brief Default destructor for MatrixProfile.
             * 
             * This destructor is defaulted and does not perform any additional actions.
             */
            ~MatrixProfile()=default;

            /**
             * @brief Update the feature group with new data.
             * 
             * This method updates the feature group with new data from the observed dataset.
             * It is expected that each derived class will implement this method to perform
             * the appropriate updates.
             * 
             * @param observed_dataset The dataset to use for the update.
             * @param window_size The size of the window to use for the update.
             */
            void update(const std::shared_ptr<DataSet>& observed_dataset, int window_size) override;
    };

    /**
     * @brief Interface for QT data.
     * 
     * This class serves as an interface for QT data, providing a pointer to the QT data array.
     * It is intended to be used as a base class for specific implementations of QT data.
     */
    class QTDataInterface{

        public:
            double* d_QT=nullptr; /**< Pointer to the QT data array */

            /**
             * @brief Default constructor for QTDataInterface.
             * 
             * This constructor initializes the QT data interface with a null pointer for the QT data array.
             */
            ~QTDataInterface()=default;
    };

    /**
     * @brief Implementation of QT data for CUDA.
     * 
     * This class provides a CUDA-specific implementation of the QT data interface.
     */
    class QTDataCUDA: public QTDataInterface{

        public:

            /**
             * @brief Default constructor for QTDataCUDA.
             * This constructor initializes the QT data CUDA object with default values.
             * It is intended to be used when no specific parameters are provided.
             */
            QTDataCUDA()=default; 

            /**
             * @brief Parameterized constructor for QTDataCUDA.
             * 
             * This constructor initializes the QT data CUDA object with the provided dataset, means, and standard deviations.
             * It is intended to be used when specific parameters are provided for the QT data.
             * 
             * @param observed_dataset The dataset to use for the QT data.
             * @param means The means of the dataset.
             * @param stds The standard deviations of the dataset.
             * @param window_size The size of the window to use for the QT data.
             * @param start_loc The starting location for the QT data.
             */
            QTDataCUDA(const std::shared_ptr<DataSet> observed_dataset,
                                const std::shared_ptr<VectorGroup<double>> means,
                                const std::shared_ptr<VectorGroup<double>> stds,
                                int window_size,
                                int start_loc);

            /**
             * @brief Default destructor for QTDataCUDA.
             * 
             * This destructor cleans up the QT data CUDA object, releasing any resources allocated during its lifetime.
             */
            ~QTDataCUDA();
    };
   



} // namespace cfade