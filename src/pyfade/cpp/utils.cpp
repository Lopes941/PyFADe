
#include <cpp/utils.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace cfade{

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
