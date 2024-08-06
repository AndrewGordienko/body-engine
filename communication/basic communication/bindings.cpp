#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Add this line for STL conversions
#include "customantenv.h"

namespace py = pybind11;

PYBIND11_MODULE(mujoco_renderer, m) {
    py::class_<CustomAntEnv>(m, "CustomAntEnv")
        .def(py::init<const char*>())
        .def("getReward", &CustomAntEnv::getReward)
        .def("getObservation", &CustomAntEnv::getObservation)
        .def("setAction", &CustomAntEnv::setAction)
        .def("render", &CustomAntEnv::render)
        .def("should_close", &CustomAntEnv::should_close);
}
