#include <Python.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "GLFWWindowManager.h"

// Custom error callback function (not used anymore)
void mujocoErrorCallback(const char* msg) {
    std::cerr << "MuJoCo Error: " << msg << std::endl;
}

static PyObject* myenv_main(PyObject* self, PyObject* args) {
    std::cout << "Starting myenv_main" << std::endl;
    // Initialize GLFW and create a window
    GLFWWindowManager windowManager;
    if (!windowManager.initialize()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize GLFW");
        return NULL;
    }

    std::cout << "GLFW initialized" << std::endl;
    windowManager.createWindow(800, 600, "MuJoCo Ant Model");

    std::cout << "Loading MuJoCo model" << std::endl;
    // Load MuJoCo model
    char error[1000] = "Could not load model";
    mjModel* m = mj_loadXML("/Users/andrewgordienko/Documents/body engine/cpp xml example/ant_model.xml", nullptr, error, 1000);
    if (!m) {
        std::cerr << error << std::endl;
        PyErr_SetString(PyExc_RuntimeError, error);
        return NULL;
    }

    std::cout << "MuJoCo model loaded" << std::endl;
    // Create MuJoCo data
    mjData* d = mj_makeData(m);
    if (!d) {
        std::cerr << "Could not create mjData" << std::endl;
        mj_deleteModel(m);
        PyErr_SetString(PyExc_RuntimeError, "Could not create mjData");
        return NULL;
    }

    std::cout << "MuJoCo data created" << std::endl;
    // Main rendering loop
    while (!windowManager.windowShouldClose()) {
        // Step the simulation
        mj_step(m, d);
        std::cout << "Simulation step" << std::endl;

        // Render the simulation
        windowManager.render(m, d);

        // Poll for and process events
        windowManager.pollEvents();
    }

    // Clean up
    mj_deleteData(d);
    mj_deleteModel(m);

    std::cout << "Cleaned up MuJoCo resources" << std::endl;
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef MyEnvMethods[] = {
    {"main", myenv_main, METH_NOARGS, "Run the main function"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef myenvmodule = {
    PyModuleDef_HEAD_INIT,
    "myenv",
    NULL,
    -1,
    MyEnvMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_myenv(void) {
    return PyModule_Create(&myenvmodule);
}
