#pragma once

#include <vector>
#include <memory>
#include <ctime>


namespace cfade
{

    template <typename T>
    struct VectorGroup
    {
        int rows;
        int cols;
        std::vector<T> data;

        VectorGroup()=default;
        VectorGroup(int);
        VectorGroup(int,int);
        VectorGroup(int,int,std::vector<T>&);

        std::vector<T> operator[](int) const;
        void increase_cols(int);

        int size()const;
        T at(int,int) const;
        T& at(int,int);
    };
    


    class IDataSetObserver{

        public:

            ~IDataSetObserver()=default;
            virtual void update()=0;
    };

    class IDataSetObservable{

        private:
        
            std::vector<std::shared_ptr<IDataSetObserver>> observers;

        protected:

            virtual void notify();

        public:

            ~IDataSetObservable()=default;

            virtual void add_observer(std::shared_ptr<IDataSetObserver>);
            virtual void remove_observer(std::shared_ptr<IDataSetObserver>);
            

    };

    class DataSet : public IDataSetObservable{

        private:

            double rate_of_reserve = 2;

            VectorGroup<double> data;

            std::vector<double> get_row_data(int) const;
            

            void check_selected_dimension(int) const;
            void check_selected_timestamp(int) const;

        public:

            DataSet()=default;
            ~DataSet()=default;
            // DataSet(VectorGroup&);
            // DataSet(std::vector<double>&);

            const VectorGroup<double>& get_data() const;
            void set_data(VectorGroup<double>&);

            // void insert_timestamp(std::vector<double>&);
            void insert_chunk(VectorGroup<double>&);

            // std::vector<double> get_data_from_dimension(int) const;
            double get_data(int, int) const;
            double at(int,int) const;
            
            int get_dimension() const;
            int get_size() const;

    };


    
} // namespace cfade

