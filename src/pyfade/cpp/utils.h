#pragma once

#include <vector>
#include <memory>
#include <variant>

namespace cfade{

    

    /**
     * @brief A 2D data container for storing data in column-major format.
     * 
     * This structure holds a representation of multi-dimensional time series
     * using a flat 'std::vector<T>' with explicit row and column metadata. It
     * provides access methods and dynamic column resizing.
     * 
     * @tparam T The type of elements stored in the vector group.
     */
    template <typename T>
    struct VectorGroup{

        using value_type = T;

        int rows; /**< Number of rows in group */
        int cols; /**< Number of columns in group */
        std::vector<T> data; /**< Vector that holds data */

        /**
         * @brief Default constructor
         * 
         * Constructs an empty VectorGroup with zero rows and columns.
         */
        VectorGroup();

        /**
         * @brief Construct a VectorGroup with specified number of rows.
         * 
         * @param rows Number of rows.
         */
        VectorGroup(int rows);

        /**
         * @brief Construct a VectorGroup with specified number of rows and columns.
         * 
         * @param rows Number of rows.
         * @param cols Number of columns.
         */
        VectorGroup(int rows,int cols);

        /**
         * @brief Construct a VectorGroup with specified number of rows and columns, and data.
         * 
         * @param rows Number of rows.
         * @param cols Number of columns.
         * @param data Flat column-major data vector. Must match rows x cols.
         */
        VectorGroup(int rows,int cols ,std::vector<T>& data);

        /**
         * @brief Access an entire row as a vector
         * 
         * @param row Row whose data must be accessed.
         * @return std::vector<T> A vector containing the values from the specified row.
         */
        std::vector<T> operator[](int row) const;

        /**
         * @brief Increase the number of columns in the group.
         * 
         * @param added_cols Number of columns to be added.
         */
        void increase_cols(int added_cols);

        /**
         * @brief Get the total number of stored elements.
         * 
         * @return int Total size = rows x cols.
         */
        int total_size()const;

        /**
         * @brief Read-only access to an element at (row,col).
         * 
         * @param row The row index.
         * @param col The column index.
         * @return T The value at given position.
         */
        T at(int row, int col) const;

        /**
         * @brief Mutable access to an element at (row,col).
         * 
         * @param row The row index.
         * @param col The column index.
         * @return T& Reference to the value at the given position.
         */
        T& at(int row, int col);
    };

    
    /**
     * @brief A variant type that can hold either an integer, double, or boolean value.
     * 
     * This is used to allow flexibility in handling different types of parameters
     * within the same structure.
     */
    using VariantTypes = std::variant<int, double, bool>;

    /**
     * @brief A variant type that can hold either a shared pointer to a VectorGroup of integers or doubles.
     * 
     * This is used to allow flexibility in handling different types of data within the same structure.
     */
    using VariantVec= std::variant<
                        std::shared_ptr<VectorGroup<int>>, 
                        std::shared_ptr<VectorGroup<double>>>;

} // namespace cfade
