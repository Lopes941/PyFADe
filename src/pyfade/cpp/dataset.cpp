
#include<cpp/dataset.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace cfade
{
    
    void DataSetObservable::add_observer(std::shared_ptr<IDataSetObserver> new_observer){

        observers.push_back(std::move(new_observer));
        
    }

    void DataSetObservable::remove_observer(std::shared_ptr<IDataSetObserver> removed_observer){

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

    void DataSetObservable::notify(){
        for(std::shared_ptr<IDataSetObserver> &o : observers){
            o->update();
        }
    }

    DataSet::DataSet() : 
        DataSetObservable(), 
        data(std::make_shared<VectorGroup<double>>(0,0)) 
    {}

    void DataSet::check_selected_dimension(int selected_dimension) const{
        if(selected_dimension > this->get_dimension()){
            throw std::runtime_error("Selected dimension greater than dimension of data");
        }
    }

    void DataSet::check_selected_timestamp(int selected_timestamp) const{
        if(selected_timestamp > this->get_length()){
            throw std::runtime_error("Selected timestamp greater than dimension of data");
        }
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

    const std::shared_ptr<VectorGroup<double>> DataSet::get_data(){
        return data;
    }

    void DataSet::set_data(const VectorGroup<double>& new_data){

        if (!data){
            data = std::make_shared<VectorGroup<double>>(new_data);
        }else{
            (*data) = new_data;
        }
        notify();
    }

    void DataSet::insert_chunk(VectorGroup<double>& new_data){

        check_selected_dimension(new_data.rows);

        int last_cols = data->cols;
        data->increase_cols(new_data.cols);
        for(int dimension=0; dimension<get_dimension();dimension++){
            for (int col=0;col<new_data.cols;col++){
                data->at(dimension,last_cols+col) = new_data.at(dimension,col);
            }
        }

        this->notify();
    }

    double DataSet::at(int selected_dimension, int selected_timestamp) const{
        return data->at(selected_dimension,selected_timestamp);
    }
    
    int DataSet::get_dimension() const{
        return data->rows;
    }

    int DataSet::get_length() const{
        return data->cols;
    }

} // namespace cfade
