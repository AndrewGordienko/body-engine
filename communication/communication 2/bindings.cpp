#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "customantenv.h"

namespace py = pybind11;

PYBIND11_MODULE(mujoco_renderer, m) {
    py::class_<CustomAntEnv>(m, "CustomAntEnv")
        .def(py::init<const char*, int>())  // Update constructor to accept max_steps
        .def("calculateReward", &CustomAntEnv::calculateReward)
        .def("getObservation", &CustomAntEnv::getObservation)
        .def("render", &CustomAntEnv::render)
        .def("should_close", &CustomAntEnv::should_close)
        .def("setAction", &CustomAntEnv::setAction)
        .def("isDone", &CustomAntEnv::isDone)  // Expose isDone method
        .def("loadNewModel", &CustomAntEnv::loadNewModel);  // Expose loadNewModel method
}
