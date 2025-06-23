#pragma once

#include <cfade/utils/utils.h>

#include <vector>
#include <memory>


namespace cfade{

    /**
     * @brief Interface for observers of a dataset.
     * 
     */
    class IDataSetObserver{

        public:

            ~IDataSetObserver()=default;
            virtual void update()=0;
    };

    /**
     * @brief Observable class for datasets.
     * 
     */
    class DataSetObservable{

        private:
        
            std::vector<std::shared_ptr<IDataSetObserver>> observers; /**< List of observers to notify on changes. */

        protected:

            /**
             * @brief Notify all observers of a change in the dataset.
             * 
             * This method will call the update method on each observer.
             */
            void notify();

        public:

            /**
             * @brief Default constructor.
             * 
             * Initializes an empty DataSetObservable.
             */
            DataSetObservable()=default;

            /**
             * @brief Default destructor.
             * 
             * Cleans up the DataSetObservable object.
             */
            ~DataSetObservable()=default;

            /**
             * @brief Add an observer to the dataset.
             * 
             * @param new_observer The observer to be added.
             */
            void add_observer(std::shared_ptr<IDataSetObserver> new_observer);

            /**
             * @brief Remove an observer from the dataset.
             * 
             * @param removed_observer The observer to be removed.
             * 
             * This method will remove the specified observer from the list of observers.
             */
            void remove_observer(std::shared_ptr<IDataSetObserver> removed_observer);
    };

    /**
     * @brief Represents a multi-dimensional time series.
     * 
     * This class holds the data for a multi-dimensional time series,
     * where each dimension is a different feature observed at the same timestamps.
     * It is derived from DataSetObservable, allowing it to notify observers of changes.
     * It can return basic information such as the number of dimensions and the length of the time series.
     * 
     */
    class DataSet : public DataSetObservable{
        
        private:
        
            std::shared_ptr<VectorGroup<double>> data; /**< Data container for the time series. */
            
            /**
             * @brief Check if the selected dimension is valid.
             * 
             * @param selected_dimension The dimension to check.
             * 
             * This method will throw an exception if the dimension is out of bounds.
             */
            void check_selected_dimension(int) const;

            /**
             * @brief Check if the selected timestamp is valid.
             * 
             * @param selected_timestamp The timestamp to check.
             * 
             * This method will throw an exception if the timestamp is out of bounds.
             */
            void check_selected_timestamp(int) const;

        public:

            /**
             * @brief Default constructor.
             * 
             * Initializes an empty DataSet with no data.
             */
            DataSet();

            /**
             * @brief Default destructor.
             * 
             * Cleans up the DataSet object.
             */
            ~DataSet()=default;

            /**
             * @brief Get the data at a specific dimension and timestamp.
             * 
             * @param selected_dimension The dimension to access.
             * @param selected_timestamp The timestamp to access.
             * @return The value at the specified dimension and timestamp.
             * 
             * This method provides read-only access to the data at the specified indices,
             * with checks to ensure the indices are valid.
             */
            double get_data(int, int) const;

            /**
             * @brief Get the data of the dataset.
             * 
             * @overload
             * @return A shared pointer to the VectorGroup containing the data.
             */
            const std::shared_ptr<VectorGroup<double>> get_data();

            /**
             * @brief Set the data of the dataset.
             * 
             * @param new_data New data to be set.
             */
            void set_data(const VectorGroup<double>&);

            /**
             * @brief Insert a chunk of data into the dataset.
             * 
             * @param new_data The new data chunk to be inserted.
             * 
             * This method will append the new data to the existing dataset and notify observers.
             */
            void insert_chunk(VectorGroup<double>&);

            
            /**
             * @brief Get the value at a specific dimension and timestamp.
             * 
             * @param int The dimension index.
             * @param int The timestamp index.
             * @return double The value at the specified dimension and timestamp.
             * 
             * This method provides read-only access to the data at the specified indices, with no checks.
             */
            double at(int,int) const;
            
            /**
             * @brief Get the number of dimensions in the dataset.
             * 
             * @return int The number of dimensions.
             * 
             * This method returns the number of dimensions (features) in the dataset.
             */
            int get_dimension() const;

            /**
             * @brief Get the length of the dataset.
             * 
             * @return int The number of timestamps in the dataset.
             * 
             * This method returns the number of timestamps (length) in the dataset.
             */
            int get_length() const;

    };


    
} // namespace cfade

