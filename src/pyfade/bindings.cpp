#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <any>
#include <variant>

#include <cfade/dataset/dataset.h>

#include <cfade/features/features.h>

#include <cfade/groups/groups.h>
#include <cfade/groups/statistics.h>
#include <cfade/groups/matrix_profile.h>
#include <cfade/groups/k_profile.h>


namespace py = pybind11;

py::array vec_to_array(const cfade::VariantVec& vec, py::object base){
    return std::visit(
        [&](auto && vec_ptr) -> py::array {

            using T = typename std::remove_reference_t<decltype(*vec_ptr)>::value_type;

            const auto& vec = *vec_ptr;
            std::vector<size_t> shape = {static_cast<size_t>(vec.rows),
                            static_cast<size_t>(vec.cols)};

            std::vector<size_t> strides = {static_cast<size_t>(sizeof(T)),
                                            static_cast<size_t>(sizeof(T)* vec.rows)};  

            return py::array_t<T>(shape,strides,vec.data.data(),base);
        }
        , vec);
}

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

class pyFeatureTrampoline: public cfade::IFeature{

    private:

        py::array_t<double> feature_vector;


    public:

        pyFeatureTrampoline(){

            py::module_ numpy = py::module_::import("numpy");

            feature_vector = numpy.attr("empty")(0);
        }

        const std::string& name() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::string&,
                cfade::IFeature,
                name
            );
        }

        cfade::VariantVec get() override{
            PYBIND11_OVERRIDE_PURE(
                cfade::VariantVec,
                cfade::IFeature,
                get
            );
        }

        py::array_t<double>& get_feature(){
            return feature_vector;
        }

};

class pyFeatureGroupTrampoline: public cfade::IFeatureGroup{

    public:


        const std::string& name() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::string&,
                cfade::IFeatureGroup,
                name
            );
        }

        const std::vector<std::string>& feature_names() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::vector<std::string>&,
                cfade::IFeatureGroup,
                feature_names
            );
        }

        const std::vector<std::string>& requirements() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::vector<std::string>&,
                cfade::IFeatureGroup,
                requirements
            );
        }

        const std::vector<std::string>& parameters() const override{
            PYBIND11_OVERRIDE_PURE(
                const std::vector<std::string>&,
                cfade::IFeatureGroup,
                parameters
            );
        }

        void set_parameter(const std::string& parameter_name, cfade::VariantTypes val) override{
            PYBIND11_OVERRIDE(
                void,
                cfade::IFeatureGroup,
                set_parameter,
                parameter_name,
                val
            );
        }
        
        void update(const std::shared_ptr<cfade::DataSet>& observed_dataset, int window_size) override{
            PYBIND11_OVERRIDE_PURE(
                void,
                cfade::IFeatureGroup,
                update,
                observed_dataset,
                window_size
            );
        }

};

class pyFeatureGroupParameterTrampoline: public cfade::IFeatureGroupParameter{

    const std::string& name()const override{
        PYBIND11_OVERRIDE_PURE(
            const std::string&,
            cfade::IFeatureGroupParameter,
            name
        );
    }

    void set(cfade::VariantTypes val) override{
        PYBIND11_OVERRIDE_PURE(
            void,
            cfade::IFeatureGroupParameter,
            set,
            val
        );
    }

};

PYBIND11_MODULE(core, handle) {
    handle.doc() = "A time-series window calculation module";

    
    {py::class_<cfade::DataSet,std::shared_ptr<cfade::DataSet>>(handle,"DataSet")
        .def(py::init<>())

        .def_property("data", 
            [](cfade::DataSet &self) -> py::array_t<double> {
                return vec_to_array(self.get_data(),py::cast(&self));
            },
            [](cfade::DataSet &self, const  py::array_t<double> arr){
                cfade::VectorGroup<double> new_vec = create_vec_from_array(arr);
                self.set_data(new_vec);
                }
        )


        .def_property_readonly("len", &cfade::DataSet::get_length)

        .def_property_readonly("ndim", &cfade::DataSet::get_dimension)

        .def_property_readonly("shape", [](const cfade::DataSet &self){
                                    int len = self.get_length();
                                    int dim = self.get_dimension();
                                    return py::make_tuple(dim,len);
                                })

        .def("insert_data", [](cfade::DataSet &self, const py::array_t<double>& new_data){
                                cfade::VectorGroup<double> new_chunk = create_vec_from_array(new_data);
                                self.insert_chunk(new_chunk);
                                })

        .def("add_observer", &cfade::DataSet::add_observer)
        .def("remove_observer", &cfade::DataSet::remove_observer)
        ;}

    {py::class_<cfade::IDataSetObserver,pyIDataFrameObserverTrampoline,std::shared_ptr<cfade::IDataSetObserver>>
        (handle,"IDataSetObserver")
        
        .def(py::init<>())

        .def("update", &cfade::IDataSetObserver::update)
        ;
    }
    
    {py::class_<cfade::IFeature,pyFeatureTrampoline,std::shared_ptr<cfade::IFeature>>
        (handle,"IFeature")

        .def(py::init<>())

        .def_property("feature",
                [](cfade::IFeature& self) -> py::array_t<double> {
                    auto* ptr = dynamic_cast<pyFeatureTrampoline*>(&self);
                    if (!ptr) throw std::runtime_error("Invalid type: expected pyFeatureTrampoline");
                    return ptr->get_feature();  // returns by reference, OK since py::array_t is ref-counted
                },

                // Setter
                [](cfade::IFeature& self, const py::array_t<double>& new_array) {
                    auto* ptr = dynamic_cast<pyFeatureTrampoline*>(&self);
                    if (!ptr) throw std::runtime_error("Invalid type: expected pyFeatureTrampoline");
                    ptr->get_feature() = new_array;
                },
    
                py::return_value_policy::reference_internal
            )

        .def("name", 
            [](cfade::IFeature& self){
                return self.name();
            })

        .def("get",
            [](cfade::IFeature& self){
                auto* trampoline = dynamic_cast<pyFeatureTrampoline*>(&self);
                if (!trampoline) throw std::runtime_error("Invalid feature type");
                auto& arr = trampoline->get_feature();
                return arr.attr("size");
            })

        .def("increase_cols",
            [](cfade::IFeature& self, const int added_size){
                auto* trampoline = dynamic_cast<pyFeatureTrampoline*>(&self);
                if (!trampoline) throw std::runtime_error("Invalid feature type");
                auto& arr = trampoline->get_feature();
                arr.attr("resize")(arr.attr("size").cast<int>()+added_size);
            })
        ;
    }

    {py::class_<cfade::IFeatureGroup,pyFeatureGroupTrampoline,std::shared_ptr<cfade::IFeatureGroup>>
        (handle,"IFeatureGroup")

        .def(py::init<>())

        .def("get_feature", 
            [](cfade::IFeatureGroup &self, std::string& feature_name) -> py::array {
                return vec_to_array(self.get_feature(feature_name),py::cast(&self));
            })

        .def("set_parameter", &cfade::IFeatureGroup::set_parameter)

        .def("update", &cfade::IFeatureGroup::update)

        .def("name", 
            [](cfade::IFeatureGroup& self){
                    return self.name();
            })

        .def("feature_names", 
            [](cfade::IFeatureGroup& self){
                    return self.feature_names();
            })

        .def("requirements", 
            [](cfade::IFeatureGroup& self){
                return self.requirements();
            })

        .def("parameters", 
            [](cfade::IFeatureGroup& self){
                return self.parameters();
            })
        ;
    }

    {py::class_<cfade::IFeatureGroupParameter,pyFeatureGroupParameterTrampoline,std::shared_ptr<cfade::IFeatureGroupParameter>>
        (handle,"IFeatureGroupParameter")

        .def(py::init<>())

        .def("set", &cfade::IFeatureGroupParameter::set)

        .def("name", 
            [](const cfade::IFeatureGroupParameter& self){
                    return self.name();
            })
        ;

    }

    {py::class_<cfade::UseCudaParam,std::shared_ptr<cfade::UseCudaParam>>
        (handle,"UseCudaParam")
        .def(py::init<>())
        .def("set", &cfade::UseCudaParam::set)
        .def("get", &cfade::UseCudaParam::get)
        .def_static("static_name", &cfade::UseCudaParam::static_name);
    }

    {py::class_<cfade::LeftOnlyParam,std::shared_ptr<cfade::LeftOnlyParam>>
        (handle,"LeftOnlyParam")
        .def(py::init<>())
        .def("set", &cfade::LeftOnlyParam::set)
        .def("get", &cfade::LeftOnlyParam::get)
        .def_static("static_name", &cfade::LeftOnlyParam::static_name);
    }

    {py::class_<cfade::SkipStartParam,std::shared_ptr<cfade::SkipStartParam>>
        (handle,"SkipStartParam")
        .def(py::init<>())
        .def("set", &cfade::SkipStartParam::set)
        .def("get", &cfade::SkipStartParam::get)

        .def_static("static_name", &cfade::SkipStartParam::static_name);
    }

    {py::class_<cfade::ExclusionZoneRatioParam,std::shared_ptr<cfade::ExclusionZoneRatioParam>>
        (handle,"ExclusionZoneRatioParam")
        .def(py::init<>())
        .def("set", &cfade::ExclusionZoneRatioParam::set)
        .def("get", &cfade::ExclusionZoneRatioParam::get)
        .def_static("static_name", &cfade::ExclusionZoneRatioParam::static_name);
    }

    {py::class_<cfade::QuantileParam,std::shared_ptr<cfade::QuantileParam>>
        (handle,"QuantileParam")
        .def(py::init<>())
        .def("set", &cfade::QuantileParam::set)
        .def("get", &cfade::QuantileParam::get)
        .def_static("static_name", &cfade::QuantileParam::static_name);
    }

    {py::class_<cfade::ContinuousStatistics,cfade::IFeatureGroup,std::shared_ptr<cfade::ContinuousStatistics>>
        (handle,"ContinuousStatistics")

        .def(py::init<std::shared_ptr<cfade::DataSet>, 
            std::vector<std::shared_ptr<cfade::IFeatureGroup>>>())

        .def_property_readonly("means", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<double> {
                return vec_to_array(self.get_feature(cfade::ContinuousMean::static_name()),py::cast(&self));
            })

        .def_property_readonly("stds", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<double> {
                return vec_to_array(self.get_feature(cfade::ContinuousStandardDeviation::static_name()),py::cast(&self));
            })

        .def_static("static_name", &cfade::ContinuousStatistics::static_name)
        .def_static("static_feature_names", &cfade::ContinuousStatistics::static_feature_names)
        .def_static("static_requirements", &cfade::ContinuousStatistics::static_requirements)
        .def_static("static_parameters", &cfade::ContinuousStatistics::static_parameters)
        ;
    }

    {py::class_<cfade::MatrixProfile,cfade::IFeatureGroup,std::shared_ptr<cfade::MatrixProfile>>
        (handle,"MatrixProfile")

        .def(py::init<std::shared_ptr<cfade::DataSet>, 
            std::vector<std::shared_ptr<cfade::IFeatureGroup>>>())
        
        .def_property_readonly("matrix_profile", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<double> {
                return vec_to_array(self.get_feature(cfade::MatrixProfileValue::static_name()),py::cast(&self));
            })

        .def_property_readonly("index", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<int> {
                return vec_to_array(self.get_feature(cfade::MatrixProfileIndex::static_name()),py::cast(&self));
            })

        .def_static("static_name", &cfade::MatrixProfile::static_name)
        .def_static("static_feature_names", &cfade::MatrixProfile::static_feature_names)
        .def_static("static_requirements", &cfade::MatrixProfile::static_requirements)
        .def_static("static_parameters", &cfade::MatrixProfile::static_parameters)

        ;
    }

    {py::class_<cfade::KProfile,cfade::IFeatureGroup,std::shared_ptr<cfade::KProfile>>
        (handle,"KProfile")

        .def(py::init<std::shared_ptr<cfade::DataSet>, 
            std::vector<std::shared_ptr<cfade::IFeatureGroup>>>())
        
        .def_property_readonly("k_profile", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<double> {
                return vec_to_array(self.get_feature(cfade::KProfileValue::static_name()),py::cast(&self));
            })

        .def_property_readonly("index", 
            [](cfade::ContinuousStatistics &self) -> py::array_t<int> {
                return vec_to_array(self.get_feature(cfade::KProfileIndex::static_name()),py::cast(&self));
            })

        .def_static("static_name", &cfade::KProfile::static_name)
        .def_static("static_feature_names", &cfade::KProfile::static_feature_names)
        .def_static("static_requirements", &cfade::KProfile::static_requirements)
        .def_static("static_parameters", &cfade::KProfile::static_parameters)

        ;
    }

    
}