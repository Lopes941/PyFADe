
#include<cpp/dataset.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace cfade
{
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
        data(data) {}

    template <typename T>
    std::vector<T> VectorGroup<T>::operator[](int idx) const{
        
        std::vector<T> row(cols);

        for (int i=0; i<cols; i++){
            row[i] = at(idx,i);
        }

        return row;
    }

    template <typename T>
    void VectorGroup<T>::increase_cols(int new_cols){
        data.resize(size()+new_cols*rows);
        cols += new_cols;
    }

    template <typename T>
    int VectorGroup<T>::size() const{
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

    void IDataSetObservable::add_observer(std::shared_ptr<IDataSetObserver> new_observer){

        observers.push_back(std::move(new_observer));
        
    }

    void IDataSetObservable::remove_observer(std::shared_ptr<IDataSetObserver> removed_observer){

        observers.erase(
        std::remove_if(
            observers.begin(),
            observers.end(),
            [&](const std::shared_ptr<IDataSetObserver>& obs) {
                return obs == removed_observer;
            }
        ),
        observers.end()
    );

    }

    void IDataSetObservable::notify(){
        for(std::shared_ptr<IDataSetObserver> &o : observers){
            o->update();
        }
    }

    
    const VectorGroup<double>& DataSet::get_data() const{
        return data;
    }

    void DataSet::set_data(VectorGroup<double>& new_data){
        data = new_data;
        notify();
    }

    void DataSet::insert_chunk(VectorGroup<double>& new_data){

        check_selected_dimension(new_data.rows);

        int last_cols = data.cols;
        data.increase_cols(new_data.cols);
        for(int dimension=0; dimension<get_dimension();dimension++){
            for (int col=0;col<new_data.cols;col++){
                data.at(dimension,last_cols+col) = new_data.at(dimension,col);
            }
        }

        this->notify();
    }

    double DataSet::get_data(int selected_dimension, int selected_timestamp)const{
        try
        {
            check_selected_dimension(selected_dimension);
            check_selected_timestamp(selected_timestamp);
            return at(selected_dimension,selected_timestamp);
        } 
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    void DataSet::check_selected_dimension(int selected_dimension) const{
        if(selected_dimension > this->get_dimension()){
            throw std::runtime_error("Selected dimension greater than dimension of data");
        }
    }

    void DataSet::check_selected_timestamp(int selected_timestamp) const{
        if(selected_timestamp > this->get_size()){
            throw std::runtime_error("Selected timestamp greater than dimension of data");
        }
    }


    double DataSet::at(int selected_dimension, int selected_timestamp) const{
        return data.at(selected_dimension,selected_timestamp);
    }
    
    int DataSet::get_dimension() const{
        return data.rows;
    }

    int DataSet::get_size() const{
        return data.cols;
    }


    template struct VectorGroup<double>;
    template struct VectorGroup<int>;
} // namespace cfade
