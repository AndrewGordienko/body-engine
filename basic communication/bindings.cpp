#include <pybind11/pybind11.h>
#include "customantenv.h"

namespace py = pybind11;

PYBIND11_MODULE(mujoco_renderer, m) {
    py::class_<CustomAntEnv>(m, "CustomAntEnv")
        .def(py::init<const char*>())
        .def("get_reward", &CustomAntEnv::getReward)
        .def("get_observation", &CustomAntEnv::getObservation)
        .def("render", &CustomAntEnv::render)
        .def("should_close", &CustomAntEnv::should_close);
}
