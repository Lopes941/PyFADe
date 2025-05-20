#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/mprofile.h"

namespace py = pybind11;

PYBIND11_MODULE(mat_profile, handle) {
    handle.doc() = "Matrix profile Calculation module";


    py::class_<cfade::Mat_Profile>(handle,"Mat_Profile")
        .def(py::init<const std::vector<double>>())
        .def(py::init<const std::vector<double>, size_t>())

        .def_property_readonly("means", &cfade::Mat_Profile::get_means)
        .def_property_readonly("stds", &cfade::Mat_Profile::get_stds)
        .def_property_readonly("series", &cfade::Mat_Profile::get_series)
        .def_property_readonly("mp", &cfade::Mat_Profile::get_mp)
        .def_property_readonly("ind", &cfade::Mat_Profile::get_mp_ind)

        .def("return_size", &cfade::Mat_Profile::return_size)
        .def("set_interval_size", &cfade::Mat_Profile::set_interval_size)
        .def("run_batch", &cfade::Mat_Profile::run_batch);
        
}