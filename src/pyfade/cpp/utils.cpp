
#include <cfade/utils/utils.h>

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace cfade{

    std::vector<int> argsort_descending(const double* data, const int size){

        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [data](int left, int right) -> bool {
                return data[left] > data[right];
            });

        return indices;
    }

    std::vector<double> quantile(const std::shared_ptr<VectorGroup<double>>& data, double p){

         if (data->data.empty()) {
            return {0.0}; // Or throw an exception
        }
        
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;

        // Calculate the index corresponding to the quantile
        size_t n = data->cols;
        size_t k = static_cast<size_t>(p * (n - 1));

        std::vector<double> result(data->rows);

        for(int dim=0; dim<data->rows; dim++){

            std::vector<double> row_data = (*data)[dim];
            
            std::nth_element(row_data.begin(),row_data.begin()+k,row_data.end());

            result[dim] = row_data[k];
        }

        return result;

    }

    template <typename T>
    VectorGroup<T>::VectorGroup(): 
        rows(0), 
        cols(0),
        data(std::vector<T>(0)) {}

    template <typename T>
    VectorGroup<T>::VectorGroup(int rows): 
        rows(rows), 
        cols(0),
        data(std::vector<T>(rows)) {}

    template <typename T>
    VectorGroup<T>::VectorGroup(int rows, int cols): 
        rows(rows), 
        cols(cols),
        data(std::vector<T>(rows*cols)) {}

    template <typename T>
    VectorGroup<T>::VectorGroup(int rows, int cols, std::vector<T>& data): 
        rows(rows), 
        cols(cols),
        data(data) 
        {
            if(data.size() != cols*rows){
                throw std::invalid_argument("Invalid size, input data has " + 
                    std::to_string(data.size()) + 
                    " elements, but given size was " + 
                    std::to_string(cols*rows));
            }
        }

    template <typename T>
    std::vector<T> VectorGroup<T>::operator[](int row) const{
        
        std::vector<T> row_data(cols);

        for (int i=0; i<cols; i++){
            row_data[i] = at(row,i);
        }

        return row_data;
    }

    template <typename T>
    void VectorGroup<T>::increase_cols(int added_cols){
        data.resize(total_size()+added_cols*rows);
        cols += added_cols;
    }

    template <typename T>
    int VectorGroup<T>::total_size() const{
        return rows*cols;
    }

    template <typename T>
    T VectorGroup<T>::at(int row, int col) const{
        return data[row + rows*col];
    }

    template <typename T>
    T& VectorGroup<T>::at(int row, int col){
        return data[row + rows*col];
    }

    template struct VectorGroup<double>;
    template struct VectorGroup<int>;


} // namespace cfade
