#include "customantenv.h"
#include <iostream>

CustomAntEnv::CustomAntEnv(const char* model_path) {
    // Load and compile model
    char error[1000] = "Could not load binary model";
    if (std::strlen(model_path) > 4 && !std::strcmp(model_path + std::strlen(model_path) - 4, ".mjb")) {
        m = mj_loadModel(model_path, 0);
    } else {
        m = mj_loadXML(model_path, 0, error, 1000);
    }
    if (!m) {
        mju_error("Load model error: %s", error);
    }

    // Make data
    d = mj_makeData(m);

    // Initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // Create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    button_left = false;
    button_middle = false;
    button_right = false;
    lastx = 0;
    lasty = 0;
}

CustomAntEnv::~CustomAntEnv() {
    // Free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);
}

void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
        mj_resetData(env->m, env->d);
        mj_forward(env->m, env->d);
    }
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));

    // update button state
    env->button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    env->button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    env->button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &env->lastx, &env->lasty);
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));

    // no buttons down: nothing to do
    if (!env->button_left && !env->button_middle && !env->button_right) {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - env->lastx;
    double dy = ypos - env->lasty;
    env->lastx = xpos;
    env->lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (env->button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (env->button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(env->m, action, dx / height, dy / height, &env->scn, &env->cam);
}

void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    CustomAntEnv* env = reinterpret_cast<CustomAntEnv*>(glfwGetWindowUserPointer(window));
    mjv_moveCamera(env->m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &env->scn, &env->cam);
}
