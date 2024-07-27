#include <Python.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "GLFWWindowManager.h"

// Custom error callback function
void mujocoErrorCallback(const char* msg) {
    std::cerr << "MuJoCo Error: " << msg << std::endl;
}

static PyObject* myenv_main(PyObject* self, PyObject* args) {
    // Initialize GLFW and create a window
    GLFWWindowManager windowManager;
    if (!windowManager.initialize()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize GLFW");
        return NULL;
    }

    windowManager.createWindow(800, 600, "MuJoCo Ant Model");

    // Check if mj_activate and mj_deactivate are available
    #if defined(mj_activate) && defined(mj_deactivate)
        // Initialize MuJoCo
        mj_activate("/path/to/mujoco200/bin/mjkey.txt");

        // Set MuJoCo error callback
        mjcb_error = mujocoErrorCallback;

        // Load MuJoCo model
        char error[1000] = "Could not load model";
        mjModel* m = mj_loadXML("ant_model.xml", nullptr, error, 1000);
        if (!m) {
            std::cerr << error << std::endl;
            PyErr_SetString(PyExc_RuntimeError, error);
            return NULL;
        }

        // Create MuJoCo data
        mjData* d = mj_makeData(m);

        // Main rendering loop
        while (!windowManager.windowShouldClose()) {
            // Step the simulation
            mj_step(m, d);

            // Render the simulation
            windowManager.render(m, d);

            // Poll for and process events
            windowManager.pollEvents();
        }

        // Clean up
        mj_deleteData(d);
        mj_deleteModel(m);
        mj_deactivate();
    #else
        std::cerr << "MuJoCo activation functions not available" << std::endl;
        PyErr_SetString(PyExc_RuntimeError, "MuJoCo activation functions not available");
        return NULL;
    #endif

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
