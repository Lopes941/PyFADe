#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "cpp/mprofile.h"

namespace py = pybind11;

PYBIND11_MODULE(mat_profile, handle) {
    handle.doc() = "Matrix profile Calculation module";


    py::class_<cfade::Mat_Profile>(handle,"Mat_Profile")
        .def(py::init<const std::vector<double>>())
        .def(py::init<const std::vector<double>, size_t>())

        .def_property_readonly("size", &cfade::Mat_Profile::get_size)
        .def_property_readonly("means", [](const cfade::Mat_Profile &self){
                                const std::vector<double>& vec = self.get_means();
                                return py::array(vec.size(),vec.data());
                            })

        .def_property_readonly("stds", [](const cfade::Mat_Profile &self){
                                const std::vector<double>& vec = self.get_stds();
                                return py::array(vec.size(),vec.data());
                            })


        .def_property_readonly("series", [](const cfade::Mat_Profile &self){
                                const std::vector<double>& vec = self.get_series();
                                return py::array(vec.size(),vec.data());
                            })

        .def_property_readonly("mp", [](const cfade::Mat_Profile &self){
                                const std::vector<double>& vec = self.get_mp();
                                return py::array(vec.size(),vec.data());
                            })

        .def_property_readonly("ind", [](const cfade::Mat_Profile &self){
                                const std::vector<int>& vec = self.get_mp_ind();
                                return py::array(vec.size(),vec.data());
                            })

        .def_property_readonly("qt", [](const cfade::Mat_Profile &self){
                                const std::vector<double>& vec = self.get_QT();
                                return py::array(vec.size(),vec.data());
                            })


        .def("set_interval_size", &cfade::Mat_Profile::set_interval_size)
        .def("set_exclusion_ratio", &cfade::Mat_Profile::set_exclusion_ratio)
        .def("set_start_ignore", &cfade::Mat_Profile::set_start_ignore)
        .def("set_left_only", &cfade::Mat_Profile::set_left_only)

        .def("run_batch", &cfade::Mat_Profile::run_batch);
        
}