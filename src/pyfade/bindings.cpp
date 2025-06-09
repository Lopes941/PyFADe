#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

#include <cpp/dataset.h>
#include <cpp/window.h>

namespace py = pybind11;

cfade::VectorGroup<double> create_vec_from_array(const py::array_t<double>& arr){

    py::buffer_info buf = arr.request();

    if(buf.ndim == 1){

        double* ptr = static_cast<double*>(buf.ptr);
        size_t size = static_cast<size_t>(buf.shape[0]);

        cfade::VectorGroup<double> result(1, size);

        for(size_t j=0;j<size;j++){
            result.at(0,j) = *(ptr + j);
        }
        return result;

    }else if(buf.ndim == 2){

        double* ptr = static_cast<double*>(buf.ptr);
        size_t num_dim = static_cast<size_t>(buf.shape[0]);
        size_t size = static_cast<size_t>(buf.shape[1]);

        cfade::VectorGroup<double> result(num_dim, size);

        for(size_t i=0;i<num_dim;i++){
            const double* dim_ptr = ptr + i * buf.strides[0] / sizeof(double);
            for(size_t j=0;j<size;j++){
                result.at(i,j) = *(dim_ptr + j*buf.strides[1]/sizeof(double));
            }
        }
        return result;
    }else{
        throw std::runtime_error("Array must be 1D or 2D");
    }

}


class pyIDataFrameObserverTrampoline: public cfade::IDataSetObserver{

    public:

        void update() override{
            PYBIND11_OVERRIDE_PURE(
                void,
                cfade::IDataSetObserver,
                update
            );
        }

};

class pyFeatureGroupTrampoline: public cfade::FeatureGroupInterface{

    public:

        const std::shared_ptr<cfade::VectorGroup<double>> get_feature(const std::string& feature_name) const override{
            PYBIND11_OVERRIDE_PURE(
                const std::shared_ptr<cfade::VectorGroup<double>>,
                cfade::FeatureGroupInterface,
                get_feature,
                feature_name
            );
        }

        const std::string& name() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::string&,
                cfade::FeatureGroupInterface,
                name
            );
        }

        const std::vector<std::string>& feature_names() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::vector<std::string>&,
                cfade::FeatureGroupInterface,
                feature_names
            );
        }

        void update(const std::shared_ptr<cfade::DataSet> dataset, int window_size) override{
            PYBIND11_OVERRIDE_PURE(
                void,
                cfade::FeatureGroupInterface,
                update,
                dataset,
                window_size
            );
        }

};



PYBIND11_MODULE(core, handle) {
    handle.doc() = "A time-series window calculation module";

    
    {py::class_<cfade::DataSet,std::shared_ptr<cfade::DataSet>>(handle,"DataSet")
        .def(py::init<>())

        .def_property("data", 
            [](cfade::DataSet &self) -> py::array_t<double> {
                const cfade::VectorGroup<double>& vec = self.get_data();

                std::vector<size_t> shape = {static_cast<size_t>(vec.rows),
                                static_cast<size_t>(vec.cols)};

                std::vector<size_t> strides = {static_cast<size_t>(sizeof(double)* vec.cols),
                                            static_cast<size_t>(sizeof(double))};                       


                return py::array_t<double>(
                    shape,
                    strides,
                    vec.data.data(),
                    py::cast(&self)
                    );
                },
            [](cfade::DataSet &self, const  py::array_t<double> arr){
                cfade::VectorGroup<double> new_vec = create_vec_from_array(arr);
                self.set_data(new_vec);
                }
        )


        .def_property_readonly("size", &cfade::DataSet::get_size)

        .def_property_readonly("ndim", &cfade::DataSet::get_dimension)

        .def_property_readonly("shape", [](const cfade::DataSet &self){
                                    int size = self.get_size();
                                    int dim = self.get_dimension();
                                    return py::make_tuple(dim,size);
                                })

        .def("insert_data", [](cfade::DataSet &self, const py::array_t<double>& new_data){
                                cfade::VectorGroup<double> new_chunk = create_vec_from_array(new_data);
                                self.insert_chunk(new_chunk);
                                })

        .def("add_observer", &cfade::DataSet::add_observer)

        ;}

    {py::class_<cfade::IDataSetObserver,pyIDataFrameObserverTrampoline,std::shared_ptr<cfade::IDataSetObserver>>(handle,"IDataSetObserver")
        
    
        .def(py::init<>())

        .def("update", &cfade::IDataSetObserver::update)
        ;
    }
    
    {py::class_<cfade::FeatureGroupInterface,pyFeatureGroupTrampoline,std::shared_ptr<cfade::FeatureGroupInterface>>(handle,"FeatureGroupInterface")

        .def(py::init<>())

        .def("get_feature", 
            [](cfade::FeatureGroupInterface &self, std::string& feature_name) -> py::array_t<double> {
            const std::shared_ptr<cfade::VectorGroup<double>> vec = self.get_feature(feature_name);

            std::vector<size_t> shape = {static_cast<size_t>(vec->rows),
                            static_cast<size_t>(vec->cols)};

            std::vector<size_t> strides = {static_cast<size_t>(sizeof(double)* vec->cols),
                                        static_cast<size_t>(sizeof(double))};                       


            return py::array_t<double>(
                shape,
                strides,
                vec->data.data(),
                py::cast(&self)
                );
            })

        .def("update", &cfade::FeatureGroupInterface::update)

        .def("name", 
            [](cfade::FeatureGroupInterface& self){
                return self.name();
            })

        .def("feature_names", 
            [](cfade::FeatureGroupInterface& self){
                return self.feature_names();
            })
        ;
    }

    {py::class_<cfade::ContinuousStatistics,cfade::FeatureGroupInterface,std::shared_ptr<cfade::ContinuousStatistics>>(handle,"ContinuousStatistics")

        .def(py::init<std::shared_ptr<cfade::DataSet>>())
        .def_property_readonly("means", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<double> {
            const std::shared_ptr<cfade::VectorGroup<double>> vec = self.get_means();

            std::vector<size_t> shape = {static_cast<size_t>(vec->rows),
                            static_cast<size_t>(vec->cols)};

            std::vector<size_t> strides = {static_cast<size_t>(sizeof(double)* vec->cols),
                                        static_cast<size_t>(sizeof(double))};                       


            return py::array_t<double>(
                shape,
                strides,
                vec->data.data(),
                py::cast(&self)
                );
            })
        .def_property_readonly("stds", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<double> {
            const std::shared_ptr<cfade::VectorGroup<double>> vec = self.get_stds();

            std::vector<size_t> shape = {static_cast<size_t>(vec->rows),
                            static_cast<size_t>(vec->cols)};

            std::vector<size_t> strides = {static_cast<size_t>(sizeof(double)* vec->cols),
                                        static_cast<size_t>(sizeof(double))};                       


            return py::array_t<double>(
                shape,
                strides,
                vec->data.data(),
                py::cast(&self)
                );
            })

        ;
    }

    
}